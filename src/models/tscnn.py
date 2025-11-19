import torch
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, in_ch, filters, kernels, strides,
                 pool=None, activation='relu', dilations=None):
        super().__init__()
        layers = []
        C_in = in_ch

        if dilations is None:
            dilations = [1] * len(filters)
        assert len(filters) == len(kernels) == len(strides) == len(dilations)

        for f, k, s, d in zip(filters, kernels, strides, dilations):
            pad = (k // 2) * d
            layers.append(
                nn.Conv1d(
                    C_in, f,
                    kernel_size=k,
                    stride=s,
                    padding=pad,
                    dilation=d
                )
            )
            layers.append(nn.BatchNorm1d(f))
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.1))

            if pool is not None:
                ptype, psize = pool
                if ptype == 'max':
                    layers.append(nn.MaxPool1d(psize))
                else:
                    layers.append(nn.AvgPool1d(psize))

            C_in = f

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, C) -> Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)
