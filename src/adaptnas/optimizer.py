import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.schedulers import exp_grl, cosine_grl

class AdaptNASOptimizer:
    """
    Implements bi-level optimization for AdaptNAS-Combined (Eq.16–17, NeurIPS 2020).
    - Lower-level: update w_R, h, h_d
    - Upper-level: update architecture A
    """

    def __init__(self, model, alpha=0.5, gamma=1.0, lr_inner=1e-3, lr_arch=1e-3, grl_sched='exp', device='cuda'):
        self.model = model.to(device)
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.grl_sched = grl_sched

        # Separate optimizers for lower and upper levels
        self.opt_inner = torch.optim.Adam(
            [
                p for n, p in model.named_parameters()
                if p.requires_grad and "discriminator" not in n and "arch_params" not in n
            ],
            lr=lr_inner
        )
        self.opt_d = torch.optim.Adam(
            [p for p in model.discriminator.parameters() if p.requires_grad],
            lr=lr_inner
        )
        arch_params = [p for n, p in model.named_parameters() if "arch_params" in n]
        self.opt_arch = torch.optim.Adam(arch_params, lr=lr_arch) if arch_params else None

        self.ce = nn.CrossEntropyLoss()

    def _compute_losses(self, xb, yb, xt, yt, p, yt_w=None):
        """
        - Source luôn CE.
        - Target:
            + Nếu yt=None → unsupervised → entropy minimization
                L_T = sum( w * entropy(logits_t) ) / sum(w)
            + Nếu yt != None → supervised CE + optional weight
        - Domain loss giữ nguyên.
        """
        sched = exp_grl if self.grl_sched == 'exp' else cosine_grl
        gamma_t = self.gamma * sched(p)

        xb, yb = xb.to(self.device), yb.to(self.device)
        xt = xt.to(self.device)
        if yt_w is not None:
            yt_w = yt_w.to(self.device)

        # ----- Forward -----
        logits_s, d_s = self.model(xb, lambda_gr=gamma_t)
        logits_t, d_t = self.model(xt, lambda_gr=gamma_t)

        # ----- Source CE -----
        loss_s = self.ce(logits_s, yb)

        # ----- Target branch -----
        if yt is None or (yt < 0).any():
            # UNSUPERVISED TARGET
            pt = torch.softmax(logits_t, dim=1).clamp_min(1e-8)
            ent = -(pt * torch.log(pt)).sum(dim=1)   # entropy từng mẫu  [B]

            if yt_w is not None:
                # weight cho mẫu "likely normal"
                w = yt_w.clamp_min(1e-4)
                loss_t = (w * ent).sum() / (w.sum() + 1e-8)
            else:
                loss_t = ent.mean()
        else:
            # SUPERVISED TARGET
            yt = yt.to(self.device)
            if yt_w is None:
                loss_t = self.ce(logits_t, yt)
            else:
                ce_t = nn.CrossEntropyLoss(reduction='none')(logits_t, yt)
                w = yt_w.clamp_min(0.2)
                w = w / (w.mean().detach() + 1e-8)
                loss_t = (w * ce_t).mean()

        # ----- Domain loss -----
        dlab_s = torch.zeros(d_s.size(0), dtype=torch.long, device=self.device)
        dlab_t = torch.ones(d_t.size(0), dtype=torch.long, device=self.device)
        loss_d_s = self.ce(d_s, dlab_s)
        if yt_w is not None:
            ce_t_domain = F.cross_entropy(d_t, dlab_t, reduction='none')
            w_dom = yt_w.clamp_min(0.0)
            w_dom = w_dom / (w_dom.mean().detach() + 1e-8)
            loss_d_t = (w_dom * ce_t_domain).mean()
        else:
            loss_d_t = self.ce(d_t, dlab_t)
        loss_d = loss_d_s + loss_d_t

        # ----- Combined lower-level objective -----
        loss_lower = self.alpha * (loss_s - loss_d) + (1 - self.alpha) * loss_t

        return loss_lower, loss_s, loss_t, loss_d




    def step_lower(self, xb, yb, xt, yt, p, yt_w=None):
        loss_lower, _, _, _ = self._compute_losses(xb, yb, xt, yt, p, yt_w=yt_w)
        self.opt_inner.zero_grad()
        self.opt_d.zero_grad()
        loss_lower.backward()
        self.opt_inner.step()
        self.opt_d.step()
        return loss_lower.item()


    def step_upper(self, val_loader, alpha=0.5):
        """
        Upper-level optimization: update architecture A based on validation accuracy.
        Trả về 'score' = 1 - acc để caller log acc chuẩn (không âm).
        """
        if self.opt_arch is None:
            return 0.0

        # ✅ Nếu lỡ truyền (val_loader, s_idx), tự bóc DataLoader
        if isinstance(val_loader, (list, tuple)):
            dl = None
            for obj in val_loader:
                # đặc trưng của DataLoader: có __iter__ và thuộc tính batch_size
                if hasattr(obj, "__iter__") and hasattr(obj, "batch_size"):
                    dl = obj
                    break
            if dl is None:
                raise ValueError(f"step_upper expected a DataLoader, got: {type(val_loader)}")
            val_iterable = dl
        else:
            val_iterable = val_loader

        self.model.train()
        correct = 0
        total = 0

        for batch in val_iterable:
            # batch có thể là (x,y) hoặc (x,y,...) -> lấy 2 phần tử đầu
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError(f"Validation batch too short: len={len(batch)}")
                xb, yb = batch[0], batch[1]
            elif isinstance(batch, dict):
                xb = batch.get("x") or batch.get("input") or list(batch.values())[0]
                yb = batch.get("y") or batch.get("label") or list(batch.values())[1]
            else:
                raise ValueError(f"Unexpected validation batch type: {type(batch)}")

            xb, yb = xb.to(self.device), yb.to(self.device)

            # forward
            logits, _ = self.model(xb)
            loss = self.ce(logits, yb)

            # update ONLY arch params
            self.opt_arch.zero_grad()
            loss.backward()
            self.opt_arch.step()

            # accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()

        acc = correct / max(1, total)
        return 1.0 - acc
    
    def _eval_error_on_loader(self, loader):
        """Return classification error on a labeled loader (0..1)."""
        self.model.eval()
        correct, total = 0, 0
        ce_sum = 0.0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    xb, yb = batch[0], batch[1]
                elif isinstance(batch, dict):
                    xb = batch.get("x") or list(batch.values())[0]
                    yb = batch.get("y") or list(batch.values())[1]
                else:
                    raise ValueError(f"Unexpected val batch type: {type(batch)}")
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, _ = self.model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        err = 1.0 - (correct / max(1, total))
        return err

    def step_upper_combined(self, val_src_loader, val_tgt_loader, alpha=0.5):
        """
        Update ONLY architecture params on 'hybrid' objective like AdaptNAS-Combined:
            hybrid_err = alpha * src_err + (1-alpha) * tgt_err
        Trả về dict: {'src_err':..., 'tgt_err':..., 'hybrid_err':...}
        """
        if self.opt_arch is None:
            return {'src_err': 0.0, 'tgt_err': 0.0, 'hybrid_err': 0.0}

        self.model.train()
        ce = self.ce

        # một bước update arch bằng tổng CE trên src và tgt (trộn theo alpha)
        total_loss = 0.0
        n_steps = 0

        # iterate ngắn trên từng loader (1 epoch mini). Có thể cân bằng số batch:
        it_src = iter(val_src_loader)
        it_tgt = iter(val_tgt_loader)
        n_iter = min(len(val_src_loader), len(val_tgt_loader))

        for _ in range(n_iter):
            xb_s, yb_s = next(it_src)
            xb_t, yb_t = next(it_tgt)
            xb_s, yb_s = xb_s.to(self.device), yb_s.to(self.device)
            xb_t, yb_t = xb_t.to(self.device), yb_t.to(self.device)

            logits_s, _ = self.model(xb_s)
            logits_t, _ = self.model(xb_t)
            loss = alpha * ce(logits_s, yb_s) + (1 - alpha) * ce(logits_t, yb_t)

            self.opt_arch.zero_grad()
            loss.backward()
            self.opt_arch.step()

            total_loss += loss.item()
            n_steps += 1

        # đo lỗi sau update
        src_err = self._eval_error_on_loader(val_src_loader)
        tgt_err = self._eval_error_on_loader(val_tgt_loader)
        hyb_err = alpha * src_err + (1 - alpha) * tgt_err
        return {'src_err': src_err, 'tgt_err': tgt_err, 'hybrid_err': hyb_err}

    def _target_batch_parts(self, batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                return batch[0], None
            if len(batch) == 2:
                second = batch[1]
                if isinstance(second, torch.Tensor) and second.dtype in (torch.long, torch.int64, torch.int32):
                    return batch[0], None
                return batch[0], second
            return batch[0], batch[-1]
        if isinstance(batch, dict):
            xb = batch.get("x") or batch.get("input") or list(batch.values())[0]
            wt = batch.get("w") or batch.get("weight")
            return xb, wt
        return batch, None

    def _source_batch_x(self, batch):
        if isinstance(batch, (tuple, list)):
            return batch[0]
        if isinstance(batch, dict):
            return batch.get("x") or batch.get("input") or list(batch.values())[0]
        return batch

    def _feature_compactness(self, feats):
        center = feats.mean(dim=0, keepdim=True)
        return ((feats - center) ** 2).sum(dim=1).mean()

    def _weighted_feature_gap(self, src_feats, tgt_feats, tgt_w=None):
        src_mean = src_feats.mean(dim=0)
        if tgt_w is None:
            tgt_mean = tgt_feats.mean(dim=0)
        else:
            tgt_w = tgt_w.to(tgt_feats.device).float().clamp_min(0.0)
            tgt_w = tgt_w / (tgt_w.sum().detach() + 1e-8)
            tgt_mean = (tgt_feats * tgt_w.unsqueeze(1)).sum(dim=0)
        return ((src_mean - tgt_mean) ** 2).mean()

    def _weighted_entropy(self, logits, tgt_w=None):
        pt = torch.softmax(logits, dim=1).clamp_min(1e-8)
        ent = -(pt * torch.log(pt)).sum(dim=1)
        if tgt_w is None:
            return ent.mean()
        tgt_w = tgt_w.to(logits.device).float().clamp_min(0.0)
        tgt_w = tgt_w / (tgt_w.mean().detach() + 1e-8)
        return (tgt_w * ent).mean()

    def evaluate_upper_unlabeled(self, src_loader, tgt_loader, alpha=0.5, beta_gap=1.0):
        self.model.eval()
        src_terms, tgt_terms, gap_terms = [], [], []
        with torch.no_grad():
            it_src = iter(src_loader)
            it_tgt = iter(tgt_loader)
            n_iter = min(len(src_loader), len(tgt_loader))
            for _ in range(n_iter):
                xb_s = self._source_batch_x(next(it_src)).to(self.device)
                xb_t, wt_t = self._target_batch_parts(next(it_tgt))
                xb_t = xb_t.to(self.device)
                if wt_t is not None:
                    wt_t = wt_t.to(self.device)

                fs = self.model.forward_features(xb_s)
                ft = self.model.forward_features(xb_t)
                logits_t, _ = self.model(xb_t, lambda_gr=0.0)

                src_terms.append(float(self._feature_compactness(fs).cpu().item()))
                tgt_terms.append(float(self._weighted_entropy(logits_t, wt_t).cpu().item()))
                gap_terms.append(float(self._weighted_feature_gap(fs, ft, wt_t).cpu().item()))

        src_obj = float(sum(src_terms) / max(1, len(src_terms)))
        tgt_obj = float(sum(tgt_terms) / max(1, len(tgt_terms)))
        gap_obj = float(sum(gap_terms) / max(1, len(gap_terms)))
        upper_obj = alpha * src_obj + (1.0 - alpha) * tgt_obj + beta_gap * gap_obj
        return {
            'src_obj': src_obj,
            'tgt_obj': tgt_obj,
            'gap_obj': gap_obj,
            'upper_obj': float(upper_obj),
        }

    def step_upper_unlabeled(self, src_loader, tgt_loader, alpha=0.5, beta_gap=1.0):
        """
        Update ONLY architecture params with an unlabeled upper objective:
          - source holdout compactness
          - weighted target entropy
          - weighted source/target feature-gap alignment
        """
        if self.opt_arch is None:
            return {'src_obj': 0.0, 'tgt_obj': 0.0, 'gap_obj': 0.0, 'upper_obj': 0.0}

        self.model.train()
        src_terms, tgt_terms, gap_terms, upper_terms = [], [], [], []

        it_src = iter(src_loader)
        it_tgt = iter(tgt_loader)
        n_iter = min(len(src_loader), len(tgt_loader))

        for _ in range(n_iter):
            xb_s = self._source_batch_x(next(it_src)).to(self.device)
            xb_t, wt_t = self._target_batch_parts(next(it_tgt))
            xb_t = xb_t.to(self.device)
            if wt_t is not None:
                wt_t = wt_t.to(self.device)

            fs = self.model.forward_features(xb_s)
            ft = self.model.forward_features(xb_t)
            logits_t, _ = self.model(xb_t, lambda_gr=0.0)

            src_obj = self._feature_compactness(fs)
            tgt_obj = self._weighted_entropy(logits_t, wt_t)
            gap_obj = self._weighted_feature_gap(fs, ft, wt_t)
            upper_loss = alpha * src_obj + (1.0 - alpha) * tgt_obj + beta_gap * gap_obj

            self.opt_arch.zero_grad()
            upper_loss.backward()
            self.opt_arch.step()

            src_terms.append(float(src_obj.detach().cpu().item()))
            tgt_terms.append(float(tgt_obj.detach().cpu().item()))
            gap_terms.append(float(gap_obj.detach().cpu().item()))
            upper_terms.append(float(upper_loss.detach().cpu().item()))

        return {
            'src_obj': float(sum(src_terms) / max(1, len(src_terms))),
            'tgt_obj': float(sum(tgt_terms) / max(1, len(tgt_terms))),
            'gap_obj': float(sum(gap_terms) / max(1, len(gap_terms))),
            'upper_obj': float(sum(upper_terms) / max(1, len(upper_terms))),
        }


