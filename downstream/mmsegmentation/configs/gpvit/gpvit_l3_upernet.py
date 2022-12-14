_base_ = [
    '../../configs/_base_/segformer/ade20k_repeat.py',
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_160k.py'
]

backbone_channels = 432
checkpoint_url = 'https://github.com/ChenhongyiYang/GPViT/releases/download/v0.0.1/gpvit_l3_in1k_300e.pth'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='GPViTSeg',
        arch='L3',
        out_indices=(2, 5, 8, 11),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_url, prefix='backbone.'),
        drop_path_rate=0.2,
        convert_syncbn=True,
        att_with_cp=False,
        group_with_cp=False),
    neck=None,
    decode_head=dict(
        type='UPerHead',
        in_channels=[backbone_channels, backbone_channels, backbone_channels, backbone_channels],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=backbone_channels,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
# ----
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006 * 2,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            '.pos_embed': dict(decay_mult=0.0),
            '.group_token': dict(decay_mult=0.0),
            '.dw_norm': dict(decay_mult=0.0)
        }))


lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
evaluation = dict(interval=160000, metric='mIoU', pre_eval=True)

work_dir = 'work_dirs/gpvit_l3_upernet'