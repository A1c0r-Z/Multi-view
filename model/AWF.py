import torch
import torch.nn as nn


class AdaptivelyWeightedFusion(nn.Module):
    def __init__(self, n_view: int, gamma):
        super().__init__()
        self.gamma = gamma
        self.weight = nn.Parameter(torch.empty(n_view))
        self.register_parameter("AWFweight", self.weight)

    def forward(self, x, s_mask):
        # x : [batch_size, n_view, d_model]
        # s_mask : [batch_size, n_view]

        w = (self.weight).pow(self.gamma)
        w = torch.mul(w.view(1, x.size(1)), torch.ones(x.size()[0], 1).cuda(non_blocking=True))  # w : [batch_size, n_view]
        w = torch.exp(w)
        if s_mask is not None:
            w = w.masked_fill(s_mask == 0, 0)
        ww = torch.sum(w, dim=-1).unsqueeze(1)
        w = torch.div(w, ww).unsqueeze(2)  # w : [batch_size, n_view, 1]
        x = torch.mul(x, w)
        x = torch.sum(x, dim=1)  # x : [batch_size, d_model]

        return x
