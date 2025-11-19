import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pool=2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        # input: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return x.transpose(1, 2)

class AdaptNASModel(nn.Module):
    def __init__(self, input_ch=9, n_classes=6, feat_dim=128):
        super().__init__()

        # === Search Space ===
        self.ops = nn.ModuleList([
            ConvBlock(input_ch, feat_dim, 3),
            ConvBlock(input_ch, feat_dim, 5),
            ConvBlock(input_ch, feat_dim, 7),
            ConvBlock(input_ch, feat_dim, 3, pool=4),
            TransformerBlock(d_model=feat_dim, nhead=8)
        ])

        # === Architecture parameters (A) ===
        self.arch_params = nn.Parameter(torch.randn(len(self.ops)))

        # === Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

        # === Domain Discriminator ===
        self.discriminator = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, lambda_gr=1.0):
        """
        Forward pass with architecture weighting and GRL.
        x: (B, C, T)
        lambda_gr: gradient reversal strength for domain adaptation.
        """
        # === Weighted mixed operation ===
        alphas = F.softmax(self.arch_params, dim=0)
        feats = 0
        for i, op in enumerate(self.ops):
            feats = feats + alphas[i] * op(x)

        # Global average pooling → (B, feat_dim)
        feats = feats.mean(dim=-1)

        # === Classifier output ===
        logits = self.classifier(feats)

        # === Domain discriminator (with GRL) ===
        rev_feats = GradReverse.apply(feats, lambda_gr)
        d_out = self.discriminator(rev_feats)

        return logits, d_out


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None
