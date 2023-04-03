import torch
from torch import nn


class LabelGuidedGraphConstraint(nn.Module):
    def __init__(self, alpha, s_mask, l_mask):
        super().__init__()
        self.alpha = alpha
        self.s_mask = s_mask
        self.l_mask = l_mask

    def forward(self, x, y):
        # x : [batch_size, n_view, d_model]
        # y : [batch_size, n_label]
        # s_mask : [batch_size, n_view]
        # l_mask : [batch_size, n_label]
        batch_size, n_view, _ = x.size()

        if self.s_mask is not None:
            s_mask = self.s_mask.transpose(0, 1).view(n_view, batch_size, 1)
            s_mask = torch.bmm(s_mask, s_mask.transpose(1, 2))  # s_mask : [n_view, batch_size, batch_size]
            s_mask = s_mask.masked_fill(torch.eye(batch_size) == 1, 0)
        else:
            s_mask = torch.ones([x.size(1), x.size(0), x.size(0)])

        if self.l_mask is not None:
            t = torch.mm(y * self.l_mask, (y * self.l_mask).transpose(0, 1))
            gg = torch.mm(self.l_mask, self.l_mask.transpose(0, 1))
        else:
            t = torch.mm(y, y.transpose(0, 1))
            gg = torch.mm(torch.ones_like(y), torch.ones_like(y).transpose(0, 1))

        t = torch.div(t, gg)  # t : [batch_size, batch_size]

        x = x.transpose(0, 1)  # x : [n_view, batch_size, d_model]
        s = (self.nxn_cos_sim(x, x) + 1.)/2.  # s : [n_view, batch_size, batch_size]

        s = s * (1 - 1e-5) + 0.5e-5  # avoid log0

        l = - torch.mul(t, torch.log(s)) - torch.mul(1-t, torch.log(1-s))  # l : [n_view, batch_size, batch_size]
        l = (l.masked_fill(torch.eye(batch_size).cuda(non_blocking=True) == 1, 0))

        l = l.masked_fill(s_mask.cuda(non_blocking=True) == 0, 0)
        l = torch.sum(l)/(2 * n_view * torch.sum(s_mask)) * self.alpha

        return l

    def nxn_cos_sim(self, A, B, dim=-1, eps=1e-8):
        numerator = torch.bmm(A, B.transpose(1, 2))
        A_l2 = torch.mul(A, A).sum(axis=dim)
        B_l2 = torch.mul(B, B).sum(axis=dim)
        denominator = torch.max(torch.sqrt(torch.mul(A_l2, B_l2)), torch.tensor(eps))
        return torch.div(numerator, denominator.unsqueeze(-1))
