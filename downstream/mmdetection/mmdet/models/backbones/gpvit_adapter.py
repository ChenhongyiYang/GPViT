# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from mmcls.gpvit_dev.models.backbones.gpvit import GPViT, resize_pos_embed
from .adapter_modules import SpatialPriorModule, InteractionBlock, get_reference_points, MSDeformAttn

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class GPViTAdapter(GPViT):
    def __init__(self,
                 pretrain_size=224,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=None,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 use_extra_extractor=True,
                 att_with_cp=False,
                 group_with_cp=False,
                 *args,
                 **kwargs):

        self.att_with_cp = att_with_cp
        self.group_with_cp = group_with_cp

        super().__init__(*args, **kwargs)

        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dims

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock_GPViT(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=self.drop_path_rate,
                # norm_layer=self.norm1,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                down_stride=8
            )
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.ad_norm1 = nn.SyncBatchNorm(embed_dim)
        self.ad_norm2 = nn.SyncBatchNorm(embed_dim)
        self.ad_norm3 = nn.SyncBatchNorm(embed_dim)
        self.ad_norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)  # s4, s8, s16, s32
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        H, W = patch_resolution
        bs, n, dim = x.shape
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, patch_resolution)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 4, W // 4).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x2 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x2, scale_factor=0.25, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.ad_norm1(c1)
        f2 = self.ad_norm2(c2)
        f3 = self.ad_norm3(c3)
        f4 = self.ad_norm4(c4)
        return [f1, f2, f3, f4]


@BACKBONES.register_module()
class GPViTAdapterSingleStage(GPViTAdapter):
    def __init__(self,
                 pretrain_size=224,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=None,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 use_extra_extractor=True,
                 att_with_cp=False,
                 group_with_cp=False,
                 *args,
                 **kwargs):
        self.att_with_cp = att_with_cp
        self.group_with_cp = group_with_cp

        super(GPViTAdapter, self).__init__(*args, **kwargs)

        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dims

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, out_c1=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock_GPViT(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=self.drop_path_rate,
                # norm_layer=self.norm1,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                down_stride=8
            )
            for i in range(len(interaction_indexes))
        ])
        self.ad_norm2 = nn.SyncBatchNorm(embed_dim)
        self.ad_norm3 = nn.SyncBatchNorm(embed_dim)
        self.ad_norm4 = nn.SyncBatchNorm(embed_dim)

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c2, c3, c4 = self.spm(x)  # s4, s8, s16, s32
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        H, W = patch_resolution
        bs, n, dim = x.shape
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, patch_resolution)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 4, W // 4).contiguous()

        if self.add_vit_feature:
            x2 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x2, scale_factor=0.25, mode='bilinear', align_corners=False)
            c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f2 = self.ad_norm2(c2)
        f3 = self.ad_norm3(c3)
        f4 = self.ad_norm4(c4)
        return [f2, f3, f4]

class InteractionBlock_GPViT(InteractionBlock):
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, patch_resolution):
        H, W = patch_resolution

        x = self.injector(query=x,
                          reference_points=deform_inputs1[0],
                          feat=c,
                          spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, patch_resolution)

        c = self.extractor(query=c,
                           reference_points=deform_inputs2[0],
                           feat=x,
                           spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2],
                           H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c,
                              reference_points=deform_inputs2[0],
                              feat=x,
                              spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2],
                              H=H, W=W)
        return x, c


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 8, w // 8)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs1, deform_inputs2
