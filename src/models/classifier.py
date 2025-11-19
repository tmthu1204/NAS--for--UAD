import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, layers, hidden, out=2):
        super().__init__()
        hs = [in_dim] + [hidden]*(layers-1) + [out]
        mods = []
        for i in range(len(hs)-1):
            mods.append(nn.Linear(hs[i], hs[i+1]))
            if i < len(hs)-2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)


    def forward(self, x):
        return self.net(x)