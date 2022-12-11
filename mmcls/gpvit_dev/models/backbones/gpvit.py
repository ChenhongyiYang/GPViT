"""
Author: Chenhongyi Yang
"""
from typing import Sequence

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer, build_conv_layer, build_activation_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks import DropPath

from mmcls.utils import get_root_logger
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed, to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone

from mmcls.gpvit_dev.models.modules.patch_embed import PatchEmbed, ConvPatchEmbed
from mmcls.gpvit_dev.models.build import build_patch_embed
from mmcls.gpvit_dev.models.utils.attentions import LePEAttnSimpleDWBlock, GPBlock

@BACKBONES.register_module()
class GPViT(BaseBackbone):
    arch_zoo = {
        **dict.fromkeys(
            ['L1', 'L1'], {
                'embed_dims': 216,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=0),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.2
            }),
        **dict.fromkeys(
            ['L2', 'L2'], {
                'embed_dims': 348,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=1),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.2
            }),
        **dict.fromkeys(
            ['L3', 'L3'], {
                'embed_dims': 432,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=1),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.3
            }),
        **dict.fromkeys(
            ['L4', 'L4'], {
                'embed_dims': 624,
                'patch_size': 8,
                'window_size': 2,
                'num_layers': 12,
                'num_heads': 12,
                'num_group_heads': 6,
                'num_group_forward_heads': 6,
                'num_ungroup_heads': 6,
                'ffn_ratio': 4.,
                'patch_embed': dict(type='ConvPatchEmbed', num_convs=2),
                'mlpmixer_depth': 1,
                'group_layers': {1: 64, 4: 32, 7: 32, 10: 16},
                'drop_path_rate': 0.3
            }),
    }
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
        super(GPViT, self).__init__(init_cfg)
        self.arch = arch

        if isinstance(arch, str):
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.test_cfg = test_cfg
        if kwargs.get('embed_dims', None) is not None:
            self.embed_dims = kwargs.get('embed_dims', None)
        else:
            self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn

        # set gradient checkpoint
        _att_with_cp = False
        if _att_with_cp is None:
            if not hasattr(self, "att_with_cp"):
                self.att_with_cp  = self.arch_settings['with_cp']
        else:
            self.att_with_cp = _att_with_cp
        _group_with_cp = kwargs.pop('group_with_cp', None)
        if _group_with_cp is None:
            if not hasattr(self, "group_with_cp"):
                self.group_with_cp = self.att_with_cp
        else:
            self.group_with_cp = _group_with_cp

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            patch_size=self.arch_settings['patch_size'],
            stride=self.arch_settings['patch_size'],
        )
        _patch_cfg.update(patch_cfg)
        _patch_cfg.update(self.arch_settings['patch_embed'])
        self.patch_embed = build_patch_embed(_patch_cfg)
        self.freeze_patch_embed = freeze_patch_embed

        self.patch_size = self.arch_settings['patch_size']

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches,self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        if drop_path_rate < 0:
            _drop_path_rate = self.arch_settings.get('drop_path_rate', None)
            if _drop_path_rate is None:
                raise ValueError
        else:
            _drop_path_rate = drop_path_rate

        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers

        for i in range(self.num_layers):
            _arch_settings = copy.deepcopy(self.arch_settings)
            if i not in _arch_settings['group_layers'].keys():
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    num_heads=_arch_settings['num_heads'],
                    window_size=_arch_settings['window_size'],
                    ffn_ratio=_arch_settings['ffn_ratio'],
                    drop_rate=drop_rate,
                    drop_path=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=self.att_with_cp)
                _layer_cfg.update(layer_cfgs[i])
                attn_layer = LePEAttnSimpleDWBlock(**_layer_cfg)
                self.layers.append(attn_layer)
            else:
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    depth=_arch_settings['mlpmixer_depth'],
                    num_group_heads=_arch_settings['num_group_heads'],
                    num_forward_heads=_arch_settings['num_group_forward_heads'],
                    num_ungroup_heads=_arch_settings['num_ungroup_heads'],
                    num_group_token=_arch_settings['group_layers'][i],
                    ffn_ratio=_arch_settings['ffn_ratio'],
                    drop_path=dpr[i],
                    with_cp=self.group_with_cp)
                group_layer = GPBlock(**_layer_cfg)
                self.layers.append(group_layer)
        self.final_norm = final_norm
        # assert final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm{i}', norm_layer)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(GPViT, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs.append(patch_token)
        return tuple(outs)
