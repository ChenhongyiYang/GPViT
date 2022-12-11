from mmcls.models.backbones import GPViT

from ..builder import BACKBONES



@BACKBONES.register_module()
class GPViTSeg(GPViT):
    def __init__(self,
                 arch='',
                 img_size=224,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None,
                 test_cfg=dict(vis_group=False),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 **kwargs):

        self.att_with_cp = False
        self.group_with_cp = False

        super(GPViTSeg, self).__init__(
                 arch,
                 img_size,
                 in_channels,
                 out_indices,
                 drop_rate,
                 drop_path_rate,
                 qkv_bias,
                 norm_cfg,
                 final_norm,
                 interpolate_mode,
                 patch_cfg,
                 layer_cfgs,
                 init_cfg,
                 test_cfg,
                 convert_syncbn,
                 freeze_patch_embed)

    def dummy(self):
        pass

