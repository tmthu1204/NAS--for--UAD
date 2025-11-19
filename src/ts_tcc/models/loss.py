import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    InfoNCE/NT-Xent cho batch size cố định (yêu cầu DataLoader drop_last=True).
    zis, zjs: [B, D]
    """
    def __init__(self, device, batch_size, temperature=0.5, use_cosine_similarity=True):
        super().__init__()
        self.device = device
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)

        # similarity function
        self.similarity_fn = self._cosine_sim if use_cosine_similarity else self._dot_sim

        # CE trên cột 0 (positive) vs các negative còn lại
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.mask = self._get_correlated_mask(self.batch_size, self.device)

    @staticmethod
    def _get_correlated_mask(bs: int, device):
        N = 2 * bs
        eye = torch.eye(N, device=device, dtype=torch.bool)
        l1  = eye.roll(shifts=bs,  dims=1)   # đường chéo dịch +bs
        l2  = eye.roll(shifts=-bs, dims=1)   # đường chéo dịch -bs
        # True ở các vị trí KHÔNG phải (i,i), (i,i+bs), (i,i-bs)
        mask = ~(eye | l1 | l2)
        return mask


    @staticmethod
    def _dot_sim(x: torch.Tensor, y: torch.Tensor):
        # x: [N, D], y: [M, D] -> [N, M]
        return x @ y.t()

    @staticmethod
    def _cosine_sim(x: torch.Tensor, y: torch.Tensor):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return x @ y.t()

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor):
        """
        returns: scalar loss (mean per sample)
        """
        bs = self.batch_size
        assert zis.shape[0] == zjs.shape[0] == bs, \
            f"NTXentLoss expects fixed batch_size={bs}, got {zis.shape[0]}, {zjs.shape[0]}." \
            " Hãy đặt DataLoader(drop_last=True) cho chế độ self-supervised."

        reps = torch.cat([zjs, zis], dim=0)            # [2B, D]
        sim  = self.similarity_fn(reps, reps)          # [2B, 2B]
        sim  = sim / self.temperature

        # positives (i với i+bs) và (i+bs với i)
        pos_left  = sim.diag(bs)                       # [B]
        pos_right = sim.diag(-bs)                      # [B]
        positives = torch.cat([pos_left, pos_right], dim=0).view(2 * bs, 1)  # [2B,1]

        negatives = sim[self.mask.to(sim.device)].view(2 * bs, -1)           # [2B, 2B-2]

        logits = torch.cat([positives, negatives], dim=1)                     # [2B, 1 + (2B-2)]
        labels = torch.zeros(2 * bs, dtype=torch.long, device=sim.device)     # pos ở cột 0

        loss = self.criterion(logits, labels)
        loss = loss / (2.0 * bs)  # mean per sample
        return loss
