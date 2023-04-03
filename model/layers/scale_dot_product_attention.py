import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    params: q k v mask=none
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2,3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        # [batch_size, head, length, length]

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score)
        # [batch_size, head, length, length]

        v = score @ v
        # [batch_size, head, length, d_tensor]

        return v, score
