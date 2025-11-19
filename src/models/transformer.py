import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))


    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]




class ARTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)


    def forward(self, x):
        x = self.pos(x)
        return self.enc(x)