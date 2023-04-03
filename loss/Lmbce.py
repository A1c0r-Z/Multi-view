import torch
from torch import nn


class MaskedBinaryCrossEntropy(nn.Module):
    def __init__(self, d_model, n_cls, beta, mask):
        super().__init__()
        self.beta = beta
        self.l_mask = mask
        self.linear1 = nn.Linear(d_model, n_cls)
        self.linears = [nn.Linear(d_model, 1) for _ in range(n_cls)]
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, p, y):
        # z, p : [batch_size, n_cls]

        l_mask = torch.ones_like(z)
        if self.l_mask is not None:
            l_mask = self.l_mask
            z = z.masked_fill(self.l_mask==0, 0)
            p = p.masked_fill(self.l_mask==0, 0)

        l1 = - torch.mul(y, torch.log(z)) - torch.mul(1 - y, torch.log(1 - z))
        l2 = - torch.mul(y, torch.log(p)) - torch.mul(1 - y, torch.log(1 - p))

        l = l1 + self.beta * l2
        l = torch.sum(l) / torch.sum(l_mask)

        return l