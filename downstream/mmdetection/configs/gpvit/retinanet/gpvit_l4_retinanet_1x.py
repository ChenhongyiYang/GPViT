_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

checkpoint_url = ''
embed_dims = 624
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='GPViTAdapterSingleStage',
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        arch='L4',
        drop_path_rate=0.1,
        out_indices=(11,),
        final_norm=False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_url, prefix="backbone."),
        convert_syncbn=True),
    neck=dict(
        type='FPN',
        in_channels=[embed_dims, embed_dims, embed_dims],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05,
    paramwise_cfg=dict(
    custom_keys={
        'level_embed': dict(decay_mult=0.),
        'pos_embed': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'bias': dict(decay_mult=0.),
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0),
        '.group_token': dict(decay_mult=0.0),
        '.dw_norm': dict(decay_mult=0.0)
    }))

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)

work_dir = 'work_dirs/gpvit_l4_retinanet_1x'