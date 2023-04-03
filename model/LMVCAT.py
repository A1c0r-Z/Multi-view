from torch import nn

from model.AWF import AdaptivelyWeightedFusion
from model.CFormer import CFormerBlock
from model.MLP import MLPBlock
from model.VFormer import VFormerBlock


class LMVCATModel(nn.Module):
    def __init__(self, n_view, d_vec, mlp_out, d_model, mlp_hidden, drop_prob,
                 vf_hidden, vf_head, vf_layers, awf_gamma, cf_hidden,
                 cf_head, cf_layers, cls_tokens, n_cls, s_mask, l_mask):
        super().__init__()
        self.s_mask = s_mask
        self.l_mask = l_mask
        self.mlp = MLPBlock(n_view=n_view,
                            d_vec=d_vec,
                            out=mlp_out,
                            hidden=mlp_hidden,
                            drop_prob=drop_prob)
        self.vformer = VFormerBlock(d_model=d_model,
                                    ffn_hidden=vf_hidden,
                                    n_head=vf_head,
                                    drop_prob=drop_prob,
                                    n_layers=vf_layers)
        self.awf = AdaptivelyWeightedFusion(n_view=n_view,
                                            gamma=awf_gamma)
        self.cformer = CFormerBlock(d_model=d_model,
                                    ffn_hidden=cf_hidden,
                                    n_head=cf_head,
                                    drop_prob=drop_prob,
                                    n_layers=cf_layers,
                                    cls_tokens=cls_tokens,
                                    n_cls=n_cls)

    def forward(self, x):
        x = self.mlp(x)  # x : [batch_size, n_view, d_model]
        x = self.vformer(x, self.s_mask)
        _x = x
        x = self.awf(x, self.s_mask)
        x, p = self.cformer(x)

        return _x, x, p
