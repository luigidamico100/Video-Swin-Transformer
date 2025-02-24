num_frames_per_video = 6

_base_ = [
    '../../_base_/models/swin/swin_tiny.py', '../../_base_/default_runtime.py'
]
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1),
           test_cfg=dict(max_testing_views=4),
           cls_head=dict(num_classes=2))

# model=dict(
#     backbone=dict(patch_size=(2,4,4), #pretrained='backbones/swin_tiny_patch4_window7_224.pth',
#                   drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.),
#     cls_head=dict(num_classes=2),
#     test_cfg=dict(max_testing_views=4))

model['test_cfg'] = dict(average_clips='prob')

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/home/luigi.damico/ICPR/foldtest_0/rawframes_train'
data_root_val = '/home/luigi.damico/ICPR/foldtest_0/rawframes_val'
data_root_test = '/home/luigi.damico/ICPR/foldtest_0/rawframes_test'
ann_file_train = '/home/luigi.damico/ICPR/foldtest_0/ICPR_train_list_rawframes.txt'
ann_file_val = '/home/luigi.damico/ICPR/foldtest_0/ICPR_val_list_rawframes.txt'
ann_file_test = '/home/luigi.damico/ICPR/foldtest_0/ICPR_test_list_rawframes.txt'
img_norm_cfg = dict(
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False) # Default
    mean=[31.875, 31.875, 31.875], std=[36.592, 36.592, 36.592], to_bgr=False) # Mine
    #mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False) # Paper

''' Default '''
# train_pipeline = [
#     #dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
#     dict(type='SampleFrames', clip_len=num_frames_per_video, frame_interval=1, num_clips=1),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='RandomResizedCrop'),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]

''' Mine'''
# train_pipeline = [
#     dict(type='SampleFrames', clip_len=num_frames_per_video, frame_interval=1, num_clips=1),
#     dict(type='RawFrameDecode'),
#     dict(type='Imgaug', transforms='default'),
#     dict(type='Imgaug', transforms=[dict(type='Rotate', rotate=(-10, 10))]),
#     dict(type='RandomResizedCrop', area_range=(.6, 1.), aspect_ratio_range=(0.8, 1.5)),
#     #dict(type='Resize', scale=(-1, 256)),
#     #dict(type='RandomResizedCrop'),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5, direction='horizontal'),
#     dict(type='ColorJitter', brightness=.25, contrast=.25, saturation=.0, hue=.0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]

''' Paper '''
train_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames_per_video, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop', area_range=(.99, 1.), aspect_ratio_range=(0.99, 1.01)),
    dict(type='Resize', scale=(224, 461), keep_ratio=False),
    dict(type='Imgaug', transforms='default'),
    dict(type='Imgaug', transforms=[dict(type='Rotate', rotate=(-10, 10))]),
    dict(type='Flip', flip_ratio=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=.25, contrast=.25, saturation=.0, hue=.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]


'''
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
'''

val_pipeline = [
    dict(type='SampleFrames', clip_len=num_frames_per_video, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 461)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
'''
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
'''

test_pipeline = val_pipeline

data = dict(
    videos_per_gpu=2,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_test,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 20

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(interval=5)
work_dir = './work_dirs/ICPR_RawframeDataset_swin_tiny_patch244_window877.py'
find_unused_parameters = False
load_from = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth'
#load_from = '/home/luigi.damico/Video-Swin-Transformer/backbone/swin_tiny_patch4_window7_224.pth'
#load_from = '/home/luigi.damico/Video-Swin-Transformer/backbone/swin_tiny_patch244_window877_kinetics400_1k.pth'
#load_from = 'model.backbone.pretrained=/home/luigi.damico/Video-Swin-Transformer/work_dirs/experiment_22/training/testfold_0/epoch_16.pth'


# do not use mmdet version fp16
#fp16 = None
#optimizer_config = dict(
#    type="DistOptimizerHook",
#    update_interval=4,
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,
#    use_fp16=True,
#)
