_base_ = [
    '../_base_/segformer/segformer.py',
    '../_base_/segformer/ade20k_repeat.py',
    '../_base_/segformer/default_runtime.py',
    '../_base_/segformer/schedule_160k_adamw.py'
]

backbone_channels = 432
checkpoint_url = 'https://github.com/ChenhongyiYang/GPViT/releases/download/v0.0.1/gpvit_l3_in1k_300e.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='GPViTSeg',
        arch='L3',
        drop_path_rate=0.2,
        out_indices=(2, 5, 8, 11),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_url, prefix='backbone.'),
        convert_syncbn=True,
        att_with_cp=False,
        group_with_cp=False),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[backbone_channels, backbone_channels, backbone_channels, backbone_channels],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006 * 2,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    constructor='CustomOptimizerConstructor',
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0),
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.),
        '.pos_embed': dict(decay_mult=0.0),
        '.group_token': dict(decay_mult=0.0),
        '.dw_norm': dict(decay_mult=0.0)})
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=2, workers_per_gpu=2)

runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)


work_dir = 'work_dirs/gpvit_l3_segformer'