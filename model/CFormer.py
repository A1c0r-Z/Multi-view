import torch
from torch import nn

from model.layers.encoder_layer import EncoderLayer


class CFormerBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, n_layers, cls_tokens, n_cls):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        self.linear1 = nn.Linear(d_model, n_cls)
        self.linears = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_cls)])
        self.sigmoid = nn.Sigmoid()
        self.cls_tokens = cls_tokens

    def forward(self, x, s_mask=None):
        # x : [batch_size, d_model]
        # cls_tokens : [n_cls, d_model]
        batch_size, d_model = x.size()
        x = x.unsqueeze(1)  # x : [batch_size, 1, d_model]
        c = self.cls_tokens.unsqueeze(0)  # c : [1, n_cls, d_model]
        c = c.repeat(batch_size, 1, 1).cuda(non_blocking=True)  # c: [batch_size, n_cls, d_model]
        x = torch.cat([x, c], dim=1)  # x : [batch_size, n_cls+1, d_model]

        for layer in self.layers:
            x = layer(x, s_mask)

        z = self.linear1(x[:, 0, :])  # z : [batch_size, n_cls]
        z = self.sigmoid(z)

        p = self.linears[0](x[:, 1, :])  # p: [batch_size, 1]
        for i in range(len(self.linears) - 1):
            _p = self.linears[i](x[:, i + 2, :])
            p = torch.cat([p, _p], dim=-1)
        p = self.sigmoid(p)

        assert p.size() == z.size()

        return z, p
