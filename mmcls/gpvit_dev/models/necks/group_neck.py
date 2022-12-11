"""
Author: Chenhongyi Yang
"""
import torch
import torch.nn as nn

from mmcv.cnn import build_norm_layer

from mmcls.gpvit_dev.models.utils.attentions import LightAttModule
from mmcls.models.builder import NECKS


@NECKS.register_module()
class GroupNeck(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_cfg=dict(type='LN')):
        super().__init__()

        self.group_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.norm_query = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = LightAttModule(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=0.,
            proj_drop=0.,
            q_project=False,
            k_project=True,
            v_project=False,
            proj_after_att=False)

    def forward(self, inputs):
        # assume inputs are normalized
        if isinstance(inputs, tuple):
            assert len(inputs) == 1
            x = inputs[0]
            if len(x.shape) == 4:
                x = x.reshape(x.size(0), x.size(1), -1).permute(0, 2, 1).contiguous()
            group_token = self.group_token.expand(x.size(0), -1, -1)
            group_token = self.norm_query(group_token)
            out = self.attn(query=group_token, key=x, value=x)
            out = out.view(out.size(0), -1)
            return (out,)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')

