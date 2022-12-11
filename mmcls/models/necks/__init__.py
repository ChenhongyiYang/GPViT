# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales

from ...gpvit_dev.models.necks.group_neck import GroupNeck

__all__ = ['GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales','GroupNeck']
