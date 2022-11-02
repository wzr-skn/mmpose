_base_ = [
    '../../../../_base_/datasets/freihand2d.py'
]

channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ])

# model settings
model = dict(
    type='TopDown',
    # backbone=dict(
    #     type='HRNet',
    #     in_channels=3,
    #     extra=dict(
    #         stage1=dict(
    #             num_modules=1,
    #             num_branches=1,
    #             block='BOTTLENECK',
    #             num_blocks=(4, ),
    #             num_channels=(64, )),
    #         stage2=dict(
    #             num_modules=1,
    #             num_branches=2,
    #             block='BASIC',
    #             num_blocks=(4, 4),
    #             num_channels=(18, 36)),
    #         stage3=dict(
    #             num_modules=4,
    #             num_branches=3,
    #             block='BASIC',
    #             num_blocks=(4, 4, 4),
    #             num_channels=(18, 36, 72)),
    #         stage4=dict(
    #             num_modules=3,
    #             num_branches=4,
    #             block='BASIC',
    #             num_blocks=(4, 4, 4, 4),
    #             num_channels=(18, 36, 72, 144),
    #             multiscale_output=True),
    #         upsample=dict(mode='bilinear', align_corners=False))),
    # keypoint_head=dict(
    #     type='TopdownHeatmapSimpleHead',
    #     in_channels=[18, 36, 72, 144],
    #     in_index=(0, 1, 2, 3),
    #     input_transform='resize_concat',
    #     out_channels=channel_cfg['num_output_channels'],
    #     num_deconv_layers=0,
    #     extra=dict(
    #         final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1,)),
    #     loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    backbone=dict(
        type='RepVGGNet',
        stem_channels=32,
        stage_channels=(48, 48, 64, 72),
        block_per_stage=(1, 3, 6, 4),
        kernel_size=[3, 3, 3, 3],
        num_out=4,
        conv_cfg=dict(type='RepVGGConv')),
    neck=dict(
        type='FuseFPN',
        in_channels=[48, 48, 64, 72],
        out_channels=48,
        conv_cfg=dict(type='RepVGGConv')),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=48,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(
            final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1, )),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[128, 128],
    heatmap_size=[32, 32],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=0.8),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.25, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=90, scale_factor=0.3),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=0.8),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]

test_pipeline = val_pipeline

data_root = '/media/traindata/hands_datasets/freihand'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='FreiHandDataset',
        ann_file=f'{data_root}/freihand_pub_v2/freihand_train.json',
        img_prefix=f'{data_root}/freihand_pub_v2/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
        cut_hand=True),
    val=dict(
        type='FreiHandDataset',
        ann_file=f'{data_root}/freihand_pub_v2/freihand_val.json',
        img_prefix=f'{data_root}/freihand_pub_v2/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
        cut_hand=True),
    test=dict(
        type='FreiHandDataset',
        ann_file=f'{data_root}/freihand_pub_v2/freihand_test.json',
        img_prefix=f'{data_root}/freihand_pub_v2/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
        cut_hand=True),
)

checkpoint_config = dict(interval=1)

log_level = 'INFO'
# work_dir = './work_dirs/hand2d/hrnetv2_freihand2d_128cuthand'
work_dir = './work_dirs/hand2d/repvggnet_fusefpn_nearest_freihand2d_128cuthand_allrepvgg'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
evaluation = dict(interval=2, metric=['PCK', 'AUC', 'EPE'], save_best='AUC')

optimizer = dict(
    type='Adam',
    lr=20e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])