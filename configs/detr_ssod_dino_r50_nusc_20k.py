#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: Anonymous  && Anonymous@com
Date: 2025-06-24 00:25:09
Description: 
LastEditors: Anonymous
LastEditTime: 2025-06-25 10:25:04
FilePath: /Meta-Semi-DETR/configs/detr_ssod_dino_r50_nusc_20k.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-06-24 00:25:09
"""
_base_ = "detr_ssod_dino_r50_nusc.py"
data_dir = './dataset/nuscenes_coco'
caption_label_dir = 'captions_ovis'
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="NuscCocoDataset",
            ann_file="${data_dir}/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="${data_dir}/images/",
            seg_prefix="${data_dir}/${caption_label_dir}/",

        ),
        unsup=dict(
            type="NuscCocoDataset",
            ann_file="${data_dir}/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="${data_dir}/images/",
            seg_prefix="${data_dir}/${caption_label_dir}/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

semi_wrapper = dict(
    type="DinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        aug_query=False,

    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=20000)

exp_dir = './tlog_exps'
work_dir = "${exp_dir}/metasemidetr/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook',
             #  log_dir='enable'
             )
    ],
)
