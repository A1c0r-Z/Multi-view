from loss.Lgc import LabelGuidedGraphConstraint
from loss.Lmbce import MaskedBinaryCrossEntropy
from torch import nn


class Loss(nn.Module):
    def __init__(self, d_model, n_cls, alpha, beta, s_mask, l_mask):
        super().__init__()
        self.l1 = LabelGuidedGraphConstraint(alpha, s_mask=s_mask, l_mask=l_mask)
        self.l2 = MaskedBinaryCrossEntropy(d_model, n_cls, beta, l_mask)

    def forward(self, x, z, p, y):
        l = self.l1(x, y) + self.l2(z, p, y)

        return l
