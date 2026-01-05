import torch
from torch.utils.data import DataLoader
from .optimizer import AdaptNASOptimizer
from ..utils.visualization import plot_curve
import torch.nn.functional as F
import os


def rotation_alignment_loss(features, labels, classifier=None):
    if classifier is None:
        classifier = torch.nn.Linear(features.size(1), 4).to(features.device)
    logits = classifier(features.detach())
    ce = torch.nn.CrossEntropyLoss()
    return ce(logits, labels)


def train_bilevel(model, ds_source, ds_target_pseudo, val_loader, device,
                  steps=100, bs=64, alpha=0.5, gamma=1.0,
                  lr_inner=1e-3, lr_arch=1e-3, grl_sched='exp',
                  log_dir="outputs/figures", tag="train_curve",
                  use_rot_align=False,
                  use_cosine_decay=True, early_stop=True, patience=5, ckpt_path=None):

    model.to(device)
    model.train()
    os.makedirs(log_dir, exist_ok=True)
    opt = AdaptNASOptimizer(model, alpha, gamma, lr_inner, lr_arch, grl_sched, device)

    # source loader
    n_source = len(ds_source)
    bs_source = min(bs, n_source) if n_source > 0 else bs
    ls = DataLoader(ds_source, batch_size=bs_source, shuffle=True, drop_last=False)

    # target loader
    try:
        t_len = len(ds_target_pseudo)
    except TypeError:
        t_len = 0
    t_bs = min(bs, t_len) if t_len > 0 else bs
    lt = DataLoader(ds_target_pseudo, batch_size=t_bs, shuffle=True, drop_last=False)

    its, itt = iter(ls), iter(lt)

    loss_log, val_log = [], []

    rot_head = torch.nn.Linear(
        getattr(model.classifier, "input_dim", 128), 4
    ).to(device)
    rot_opt = torch.optim.Adam(rot_head.parameters(), lr=1e-3) if use_rot_align else None
    ce_rot = torch.nn.CrossEntropyLoss()

    sched_inner = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt.opt_inner, T_max=steps
    ) if use_cosine_decay else None
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt.opt_d, T_max=steps
    ) if use_cosine_decay else None

    best_val_acc = -1.0
    best_state = None
    stale = 0

    def _entropy_minimization_loss(logits, w=None):
        p = torch.softmax(logits, dim=1).clamp_min(1e-8)
        ent = -(p * torch.log(p)).sum(dim=1)  # [B]
        if w is not None:
            w = w.to(logits.device).float().clamp_min(0.0)
            w = w / (w.mean().detach() + 1e-8)
            return (w * ent).mean()
        return ent.mean()

    for step in range(steps):
        # -------- target batch --------
        try:
            xtb = next(itt)
        except StopIteration:
            itt = iter(lt)
            try:
                xtb = next(itt)
            except StopIteration:
                break

        xt, yt, yt_w = None, None, None

        if isinstance(xtb, (list, tuple)):
            if len(xtb) == 1:
                xt = xtb[0]
            elif len(xtb) == 2:
                # ✅ (x,y) hoặc (x,w)
                xt = xtb[0]
                second = xtb[1]
                if isinstance(second, torch.Tensor) and second.dtype in (torch.long, torch.int64, torch.int32):
                    yt = second
                else:
                    yt = None
                    yt_w = second
            elif len(xtb) == 3:
                xt, yt, yt_w = xtb[0], xtb[1], xtb[2]
            else:
                raise ValueError(f"Unexpected target batch format, len={len(xtb)}")

        elif isinstance(xtb, dict):
            xt = xtb.get("x") or xtb.get("input") or list(xtb.values())[0]
            yt = xtb.get("y") or xtb.get("label") or None
            yt_w = xtb.get("w") or xtb.get("weight") or None
        else:
            xt = xtb

        has_t_label = (yt is not None)

        # -------- source batch --------
        try:
            sb = next(its)
        except StopIteration:
            its = iter(ls)
            try:
                sb = next(its)
            except StopIteration:
                break

        if isinstance(sb, (list, tuple)):
            xb, yb = sb[0], sb[1]
        elif isinstance(sb, dict):
            xb = sb.get("x") or sb.get("input") or list(sb.values())[0]
            yb = sb.get("y") or sb.get("label") or list(sb.values())[1]
        else:
            xb, yb = sb

        p = (step + 1) / steps

        if not has_t_label:
            xb, yb = xb.to(device), yb.to(device)
            xt = xt.to(device)
            if yt_w is not None:
                yt_w = yt_w.to(device)

            gamma_t = gamma * float(p)
            logits_s, d_s = model(xb, lambda_gr=gamma_t)
            logits_t, d_t = model(xt, lambda_gr=gamma_t)

            loss_s = F.cross_entropy(logits_s, yb)
            loss_t = _entropy_minimization_loss(logits_t, w=yt_w)

            dlab_s = torch.zeros(d_s.size(0), dtype=torch.long, device=device)
            dlab_t = torch.ones(d_t.size(0), dtype=torch.long, device=device)
            loss_d = F.cross_entropy(d_s, dlab_s) + F.cross_entropy(d_t, dlab_t)

            loss_lower = alpha * (loss_s - loss_d) + (1 - alpha) * loss_t
        else:
            loss_lower, _, _, _ = opt._compute_losses(xb, yb, xt, yt, p, yt_w=yt_w)

        if use_rot_align:
            model.eval()
            with torch.no_grad():
                xr_list, yr_list = [], []
                for r in range(4):
                    xr = torch.rot90(xb, k=r, dims=[2, 1])
                    xr_list.append(xr)
                    yr_list.append(torch.full((xb.size(0),), r, dtype=torch.long, device=device))
                Xrot = torch.cat(xr_list, dim=0)
                Yrot = torch.cat(yr_list, dim=0)
                f_rot = model.forward_features(Xrot)
            model.train()
            logits_rot = rot_head(f_rot)
            loss_rot = ce_rot(logits_rot, Yrot)
            loss_lower = loss_lower + 0.1 * loss_rot

            if rot_opt is not None:
                rot_opt.zero_grad()
                loss_rot.backward(retain_graph=True)

        opt.opt_inner.zero_grad()
        opt.opt_d.zero_grad()
        loss_lower.backward()
        opt.opt_inner.step()
        opt.opt_d.step()

        if sched_inner:
            sched_inner.step()
        if sched_d:
            sched_d.step()

        loss_log.append(loss_lower.item())

        if (step + 1) % 20 == 0 or step == steps - 1:
            val_score = opt.step_upper(val_loader, alpha)
            val_acc = 1.0 - val_score
            val_log.append(val_acc)
            print(f"[BiLevel] step {step+1}/{steps} | train_loss={loss_lower:.4f} | val_acc={val_acc:.4f}")

            if early_stop:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    print(f"[BiLevel] Early stop at step {step+1}, best_val_acc={best_val_acc:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(best_state, ckpt_path)

    plot_curve(list(range(len(loss_log))), loss_log,
               os.path.join(log_dir, f"{tag}_loss.png"), title="Train Loss")
    if val_log:
        plot_curve(
            list(range(0, len(loss_log), max(1, len(loss_log)//len(val_log)))),
            val_log,
            os.path.join(log_dir, f"{tag}_val_acc.png"),
            title="Validation Accuracy"
        )

    return {"train_loss": loss_log, "val_acc": val_log}


def quick_validate(model, dl, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dl:
            if isinstance(batch, (list, tuple)):
                xb, yb = batch[0], batch[1]
            elif isinstance(batch, dict):
                xb = batch.get("x") or batch.get("input") or list(batch.values())[0]
                yb = batch.get("y") or batch.get("label") or list(batch.values())[1]
            else:
                xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)
