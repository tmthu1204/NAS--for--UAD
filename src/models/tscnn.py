import torch
import torch.nn as nn


class ConvStage(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, pool=None, activation='relu'):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        layers = [
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.1),
        ]

        if pool is not None:
            ptype, psize = pool
            if ptype == 'max':
                layers.append(nn.MaxPool1d(psize))
            else:
                layers.append(nn.AvgPool1d(psize))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EncoderCNN(nn.Module):
    def __init__(self, in_ch, filters, kernels, strides,
                 pool=None, activation='relu', dilations=None):
        super().__init__()
        C_in = in_ch

        if dilations is None:
            dilations = [1] * len(filters)
        assert len(filters) == len(kernels) == len(strides) == len(dilations)

        self.blocks = nn.ModuleList()
        self.out_channels = list(filters)
        for f, k, s, d in zip(filters, kernels, strides, dilations):
            self.blocks.append(
                ConvStage(
                    C_in,
                    f,
                    kernel_size=k,
                    stride=s,
                    dilation=d,
                    pool=pool,
                    activation=activation,
                )
            )
            C_in = f

    def forward(self, x, return_all=False):
        # x: (B, T, C) -> Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        outs = []
        for block in self.blocks:
            x = block(x)
            if return_all:
                outs.append(x.transpose(1, 2))

        if return_all:
            return outs
        return x.transpose(1, 2)
