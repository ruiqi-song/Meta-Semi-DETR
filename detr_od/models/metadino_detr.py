#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: Anonymous  && Anonymous@com
Date: 2025-01-02 10:02:03
Description: 
LastEditors: Anonymous
LastEditTime: 2025-06-23 09:29:01
FilePath: /MetaSemiDetr/detr_od/models/adapter_dino.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-01-02 10:02:03
"""
import torch
import warnings
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import DETECTORS


@DETECTORS.register_module()
class METADINODETR(SingleStageDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(METADINODETR, self).__init__(backbone, None, bbox_head, train_cfg,
                                           test_cfg, pretrained, init_cfg)

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_caption,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        # add the image information into the image metas
        for i in range(len(img_metas)):
            img_meta = img_metas[i]
            img_meta['img'] = img[i]
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels,  gt_bboxes_ignore, gt_caption=gt_caption, **kwargs)
        return losses

    # over-write `onnx_export` because:
    # (1) the forward of bbox_head requires img_metas
    # (2) the different behavior (e.g. construction of `masks`) between
    # torch and ONNX model, during the forward of bbox_head
    def onnx_export(self, img, img_metas):
        """Test function for exporting to ONNX, without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        # forward of this head requires img_metas
        outs = self.bbox_head.forward_onnx(x, img_metas)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels
