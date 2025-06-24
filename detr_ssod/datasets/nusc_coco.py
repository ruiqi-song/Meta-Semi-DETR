#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-06-23 23:41:18
Description: 
LastEditors: knightdby
LastEditTime: 2025-06-23 23:44:10
FilePath: /MetaSemiDetr/detr_ssod/datasets/nusc_coco.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-06-23 23:41:18
"""
from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class NuscCocoDataset(CocoDataset):
    CLASSES = ('car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')
