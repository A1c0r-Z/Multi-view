from torch import nn

from model.layers.layer_norm import LayerNorm
from model.layers.multi_head_attention import MultiHeadAttention
from model.layers.MLP_layer import MLPLayer


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.mlp = MLPLayer(d_model=d_model, out=d_model, hidden=ffn_hidden, drop_prob=drop_prob)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. mlp
        _x = x
        x = self.mlp(x)
        x = _x + x

        return x