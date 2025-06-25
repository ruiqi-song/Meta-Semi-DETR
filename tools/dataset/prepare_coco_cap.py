#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: Anonymous  && Anonymous@163.com
Date: 2025-01-09 11:20:17
Description: 
LastEditors: Anonymous
LastEditTime: 2025-06-25 14:33:07
FilePath: /Meta-Semi-DETR/tools/dataset/prepare_coco_cap.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-01-09 11:20:17
"""
json_files = ['./dataset/coco2017/annotations/captions_train2017.json',
              './dataset/coco2017/annotations/captions_val2017.json']

for js_file in json_files:
    with open(js_file, "r", encoding='utf-8') as f:
        js_data = json.load(f)
    for js in tqdm(js_data['annotations']):
        img_name = '%012d.txt' % js['image_id']
        caption = js['caption']
        save_path = osp.join('./dataset/coco2017/captions_coco', img_name)
        make_path_dirs(save_path)
        with open(save_path, 'a') as f:
            f.write(caption+'\n')
