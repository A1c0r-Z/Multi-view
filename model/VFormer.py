from torch import nn

from model.layers.encoder_layer import EncoderLayer


class VFormerBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):
        # [batch_size, n_view, d_model]
        for layer in self.layers:
            x = layer(x, s_mask)

        return x
