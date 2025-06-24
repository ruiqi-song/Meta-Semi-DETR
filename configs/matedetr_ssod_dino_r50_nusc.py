#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-06-23 22:12:46
Description: 
LastEditors: knightdby
LastEditTime: 2025-06-24 08:57:35
FilePath: /MetaSemiDetr/configs/matedetr_ssod_dino_r50_nusc.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-06-23 22:12:46
"""
_base_ = [
    "ssod_matedino_r50_nusc_12k.py",
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_caption=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                           (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333),
                           (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True
            ),
            dict(type="RandFlip", flip_ratio=0.0),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_caption'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                           (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333),
                           (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True
            ),
            dict(type="RandFlip", flip_ratio=0.0),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_caption'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                           (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333),
                           (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True
            ),
            dict(type="RandFlip", flip_ratio=0.0),
        ],
        record=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", 'gt_caption'],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_caption=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=None,
    workers_per_gpu=None,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type="NuscCocoDataset",
            ann_file=None,
            img_prefix=None,
            pipeline=train_pipeline,
        ),
        unsup=dict(
            type="NuscCocoDataset",
            ann_file=None,
            img_prefix=None,
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
            by_prob=True,
            # at_least_one=True,
            epoch_length=1670,
        )
    ),
)

semi_wrapper = dict(
    type="MetaDinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),

]
evaluation = dict(type="SubModulesDistEvalHook",
                  interval=2000, save_best='auto')
lr_config = dict(step=[12000, 16000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=18000)
checkpoint_config = dict(by_epoch=False, interval=2000,
                         max_keep_ckpts=2, create_symlink=False)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
