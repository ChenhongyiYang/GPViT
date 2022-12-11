from mmcv.utils import Registry, build_from_cfg
from torch import nn
import warnings


PATCH_EMBED = Registry('patch_embed')
GROUP_LAYER = Registry('group_layer')
ATTN_LAYER  = Registry('attn_layer')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_patch_embed(cfg):
    return build(cfg, PATCH_EMBED)

def build_group_layer(cfg):
    return build(cfg, GROUP_LAYER)

def build_attn_layer(cfg):
    return build(cfg, ATTN_LAYER)

