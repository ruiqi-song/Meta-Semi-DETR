#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-06-24 00:21:53
Description: 
LastEditors: knightdby
LastEditTime: 2025-06-24 08:58:14
FilePath: /MetaSemiDetr/configs/ssod_dino_r50_nusc_12k.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-06-24 00:21:53
"""
mmdet_base = "../3rdparty/mmdetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/datasets/coco_detection.py",
    f"{mmdet_base}/default_runtime.py",
]

model = dict(
    type='DINODETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='weights/resnet50-0676ba61.pth')),
    bbox_head=dict(
        type='DINODETRSSODHead',
        num_query=900,
        query_dim=4,
        random_refpoints_xy=False,
        bbox_embed_diff_each_layer=False,
        num_classes=10,
        in_channels=2048,
        transformer=dict(type='DINOTransformer', num_queries=900),
        positional_encoding=dict(
            type='SinePositionalEncodingHW', temperatureH=20, temperatureW=20, num_feats=128, normalize=True),
        loss_cls1=dict(
            type='TaskAlignedFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            loss_weight=2.0),
        loss_cls2=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner1=dict(
            type='O2MAssigner'),
        assigner2=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),),
        warm_up_step=10000),
    test_cfg=dict(max_per_img=300, warm_up_step=10000))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_caption=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
            [
            dict(
                type='Resize',
                img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=False),   # make sure don't allow to crop the patch without gt bbox to avoid some bug
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                           (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333),
                           (768, 1333), (800, 1333)],
                multiscale_mode='value',
                override=True,
                keep_ratio=True)
        ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_caption'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_dir = '/file_system/nas/algorithm/ruiqi.song/helios/data/nuscenes_coco'
caption_label_dir = 'captions_ovis'

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(type="NuscCocoDataset",
               ann_file="${data_dir}/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
               img_prefix="${data_dir}/images/",
               seg_prefix="${data_dir}/${caption_label_dir}/",
               pipeline=train_pipeline),
    val=dict(
        type='NuscCocoDataset',
        ann_file="${data_dir}/annotations/val.json",
        img_prefix="${data_dir}/images/",
        pipeline=test_pipeline),
    test=dict(
        type='NuscCocoDataset',
        ann_file="${data_dir}/annotations/val.json",
        img_prefix="${data_dir}/images/",
        pipeline=test_pipeline))

custom_hooks = [
    dict(type="NumClassCheckHook"),
]

checkpoint_config = dict(by_epoch=False, interval=2000,
                         create_symlink=False, max_keep_ckpts=2)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[12000, 16000])
runner = dict(type="IterBasedRunner", max_iters=18000)
find_unused_parameters = True
evaluation = dict(interval=2000, metric='bbox', save_best='auto')
work_dir = "./tlog_exps/test_coco/${cfg_name}/${percent}/${fold}"
