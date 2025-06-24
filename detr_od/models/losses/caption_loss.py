#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-01-02 11:14:52
Description: 
LastEditors: knightdby
LastEditTime: 2025-06-23 09:38:40
FilePath: /MetaSemiDetr/detr_od/models/losses/caption_loss.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-01-02 11:14:52
"""

import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class CaptionLoss(nn.Module):
    def __init__(self,
                 itc_loss_weight=0.5,
                 lm_loss_weight=0.5,
                 itm_loss_weight=0.3):
        super().__init__()
        self.itc_loss_weight = itc_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.itm_loss_weight = itm_loss_weight

    def forward(self,
                caption_loss):
        loss_itc = caption_loss.loss_itc*self.itc_loss_weight
        loss_itm = caption_loss.loss_itm*self.itm_loss_weight
        loss_lm = caption_loss.loss_lm*self.lm_loss_weight
        return loss_itc, loss_itm, loss_lm
