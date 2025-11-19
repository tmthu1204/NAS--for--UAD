import torch
from torch.utils.data import DataLoader
from .optimizer import AdaptNASOptimizer
from ..utils.visualization import plot_curve
import torch.nn.functional as F
import os


# ============================================================
# Rotation Alignment Loss (auxiliary self-supervised task)
# ============================================================
def rotation_alignment_loss(features, labels, classifier=None):
    """
    Compute self-supervised rotation loss (0°,90°,180°,270° alignment).
    Args:
        features: latent representations [B, D]
        labels: integer rotation labels [B] ∈ {0,1,2,3}
        classifier: optional small head to predict rotation
    """
    if classifier is None:
        # Default small linear classifier if none is provided
        classifier = torch.nn.Linear(features.size(1), 4).to(features.device)

    logits = classifier(features.detach())   # freeze encoder gradient
    ce = torch.nn.CrossEntropyLoss()
    return ce(logits, labels)


# ============================================================
# Bi-level training
# ============================================================
def train_bilevel(model, ds_source, ds_target_pseudo, val_loader, device,
                  steps=100, bs=64, alpha=0.5, gamma=1.0,
                  lr_inner=1e-3, lr_arch=1e-3, grl_sched='exp',
                  log_dir="outputs/figures", tag="train_curve",
                  use_rot_align=False,
                  use_cosine_decay=True, early_stop=True, patience=5, ckpt_path=None):
    """
    Two-phase AdaptNAS training with live logging.
    - Hỗ trợ batch target dạng (x, y) hoặc (x, y, w).
    - Cosine LR decay tùy chọn.
    - Early stopping theo val_acc, khôi phục checkpoint tốt nhất.
    """
    model.to(device)
    os.makedirs(log_dir, exist_ok=True)
    opt = AdaptNASOptimizer(model, alpha, gamma, lr_inner, lr_arch, grl_sched, device)

    # ====== SOURCE DATALOADER ======
    n_source = len(ds_source)
    bs_source = min(bs, n_source) if n_source > 0 else bs
    ls = DataLoader(ds_source, batch_size=bs_source, shuffle=True, drop_last=False)

    # ====== TARGET DATALOADER ======
    try:
        t_len = len(ds_target_pseudo)
    except TypeError:
        t_len = 0
    t_bs = min(bs, t_len) if t_len > 0 else bs
    lt = DataLoader(ds_target_pseudo, batch_size=t_bs, shuffle=True, drop_last=False)

    its, itt = iter(ls), iter(lt)

    loss_log, val_log = [], []

    # optional rotation classifier
    rot_head = torch.nn.Linear(
        getattr(model.classifier, "input_dim", 128), 4
    ).to(device)
    rot_opt = torch.optim.Adam(rot_head.parameters(), lr=1e-3) if use_rot_align else None
    ce_rot = torch.nn.CrossEntropyLoss()

    # schedulers (tuỳ chọn)
    sched_inner = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt.opt_inner, T_max=steps
    ) if use_cosine_decay else None
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt.opt_d, T_max=steps
    ) if use_cosine_decay else None

    # early stopping state
    best_val_acc = -1.0
    best_state = None
    stale = 0

    def _entropy_minimization_loss(logits):
        p = torch.softmax(logits, dim=1).clamp_min(1e-8)
        ent = -(p * torch.log(p)).sum(dim=1)
        return ent.mean()

    for step in range(steps):
        # ----- lấy batch target (có thể unlabeled hoặc (x,y[,w])) -----
        try:
            xtb = next(itt)
        except StopIteration:
            itt = iter(lt)
            try:
                xtb = next(itt)
            except StopIteration:
                # không còn batch target nào -> dừng hẳn
                break

        xt, yt, yt_w = None, None, None
        if isinstance(xtb, (list, tuple)):
            if len(xtb) == 1:
                xt = xtb[0]
            elif len(xtb) == 2:
                xt, yt = xtb
            else:
                xt, yt, yt_w = xtb[0], xtb[1], xtb[2]
        elif isinstance(xtb, dict):
            xt = xtb.get("x") or xtb.get("input") or list(xtb.values())[0]
            yt = xtb.get("y") or xtb.get("label") or None
        else:
            xt = xtb

        # ----- lấy batch source -----
        try:
            xb, yb = next(its)
        except StopIteration:
            its = iter(ls)
            try:
                xb, yb = next(its)
            except StopIteration:
                # không còn source -> dừng
                break

        # ----- lower-level update -----
        p = (step + 1) / steps

        if yt is None:
            xb, yb = xb.to(device), yb.to(device)
            xt = xt.to(device)
            gamma_t = gamma * (p if hasattr(p, "__float__") else 1.0)
            logits_s, d_s = model(xb, lambda_gr=gamma_t)
            logits_t, d_t = model(xt, lambda_gr=gamma_t)

            loss_s = F.cross_entropy(logits_s, yb)
            loss_t = _entropy_minimization_loss(logits_t)

            dlab_s = torch.zeros(d_s.size(0), dtype=torch.long, device=device)
            dlab_t = torch.ones(d_t.size(0), dtype=torch.long, device=device)
            loss_d = F.cross_entropy(d_s, dlab_s) + F.cross_entropy(d_t, dlab_t)

            loss_lower = alpha * (loss_s - loss_d) + (1 - alpha) * loss_t
        else:
            loss_lower, _, _, _ = opt._compute_losses(xb, yb, xt, yt, p, yt_w=yt_w)

        # --- Rotation alignment (optional) ---
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

        # ----- validation mỗi 20 bước -----
        if (step + 1) % 20 == 0 or step == steps - 1:
            val_score = opt.step_upper(val_loader, alpha)
            val_acc = 1.0 - val_score
            val_log.append(val_acc)
            print(f"[BiLevel] step {step+1}/{steps} | train_loss={loss_lower:.4f} | val_acc={val_acc:.4f}")

            # early stopping
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

    # khôi phục model tốt nhất & lưu checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(best_state, ckpt_path)

    # vẽ curves
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



# ============================================================
# Quick validation
# ============================================================
def quick_validate(model, dl, device):
    """Quick validation accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)
