#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v1.0.0
Author: Anonymous  && Anonymous@163.com
Date: 2025-01-25 21:48:08
Description: 
LastEditors: Anonymous
LastEditTime: 2025-06-25 14:29:59
FilePath: /Meta-Semi-DETR/tools/dataset/export_nusc2coco.py
Copyright 2025  by Inc, All Rights Reserved. 
2025-01-25 21:48:08
"""
from manifast import *
from collections import OrderedDict
from typing import List, Tuple, Union
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.splits import create_splits_scenes


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    return repro_rec


def get_2d_boxes(sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token)
                for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (
        ann_rec['visibility_token'] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(
            corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

            det_name = category_to_detection_name(box.name)

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(
            ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


def get_scene_tokens_by_name(nusc, scene_names):
    return [scene['token'] for scene in nusc.scene if scene['name'] in scene_names]


def get_camera_sample_data_tokens(nusc, scene_tokens):
    camera_tokens = []
    camera_channels = [
        'CAM_FRONT',
        # 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        # 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]

    for scene_token in scene_tokens:
        scene = nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']

        while sample_token:
            sample = nusc.get('sample', sample_token)
            for channel in camera_channels:
                cam_data_token = sample['data'][channel]
                camera_tokens.append(cam_data_token)
            sample_token = sample['next']
    return camera_tokens


SPLITS = {
    'train': 'v1.0-trainval',
    'val': 'v1.0-trainval',
    # 'mini_val': 'v1.0-mini',
    # 'mini_train': 'v1.0-mini',
    # 'test': 'v1.0-test',
}
DATA_PATH = './dataset/nuscenes_coco/nuscenes'
OUT_PATH = './dataset/nuscenes_coco'

CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

SENSOR_ID = {'RADAR_FRONT': 7, 'RADAR_FRONT_LEFT': 9,
             'RADAR_FRONT_RIGHT': 10, 'RADAR_BACK_LEFT': 11,
             'RADAR_BACK_RIGHT': 12,  'LIDAR_TOP': 8,
             'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2,
             'CAM_BACK_RIGHT': 3, 'CAM_BACK': 4, 'CAM_BACK_LEFT': 5,
             'CAM_FRONT_LEFT': 6}


CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}
categories_info = [{'name': CATS[i], 'id': i + 1}
                   for i in range(len(CATS))]
USED_CAMERA = 'CAM_FRONT'
# visibilities = ['', '1', '2', '3', '4']
visibilities = ['2', '3', '4']
make_path_dirs(OUT_PATH)
split = 'train'
data_path = DATA_PATH
nusc = NuScenes(
    version=SPLITS[split], dataroot=data_path, verbose=True)
splits = create_splits_scenes()
train_scene_names = splits['train']
val_scene_names = splits['val']
train_scene_tokens = get_scene_tokens_by_name(nusc, train_scene_names)
val_scene_tokens = get_scene_tokens_by_name(nusc, val_scene_names)
train_camera_tokens = get_camera_sample_data_tokens(nusc, train_scene_tokens)
val_camera_tokens = get_camera_sample_data_tokens(nusc, val_scene_tokens)


def main(split='train'):
    sample_data_camera_tokens = None
    # sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera') and
    #                              s['is_key_frame'] and (s['channel'] == USED_CAMERA)]
    if split == 'train':
        sample_data_camera_tokens = train_camera_tokens
    else:
        sample_data_camera_tokens = val_camera_tokens

    out_path = osp.join(OUT_PATH, 'annotations', '{}.json'.format(split))
    make_path_dirs(out_path)
    categories_info = [{'name': CATS[i], 'id': i + 1}
                       for i in range(len(CATS))]
    annotation_ret = {'categories': categories_info,
                      'annotations': [], 'images': []}
    ann_id_cnt = 0
    for sample_data_token in tqdm(sample_data_camera_tokens):
        reprojection_records = get_2d_boxes(
            sample_data_token, visibilities)
        if not reprojection_records:
            continue
        nu_info = reprojection_records[0]
        ids = nu_info['filename'].split(
            '/')[-1].split('-')[0]+'-'+nu_info['filename'].split('__')[-1][:-4]
        img_path = osp.join(DATA_PATH, nu_info['filename'])
        if not osp.exists(img_path):
            continue
        img_save_path = osp.join(
            OUT_PATH, 'images', '{}.jpg'.format(ids))
        make_path_dirs(img_save_path)
        shutil.copy(img_path, img_save_path)
        annotation_ret['images'].append({'file_name': f'{ids}.jpg',
                                        'id': ids,
                                         'width': 1600,
                                         'height': 900})
        for nu_info in reprojection_records:
            category_name = category_to_detection_name(
                nu_info['category_name'])
            if not category_name:
                continue
            category_id = CAT_IDS[category_name]
            x1, y1, x2, y2 = nu_info['bbox_corners']
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            area = width * height
            if width > 0 and height > 0:
                annotation_ret['annotations'].append({
                    'area': area,
                    'category': category_name,
                    'category_id': category_id,
                    'iscrowd': 0,
                    'id': ann_id_cnt,
                    'image_id': ids,
                    'bbox': [x1, y1, width, height],
                    # 'segmentation': segmentation
                })
                ann_id_cnt += 1
        break

    with open(out_path, 'w') as fh:
        json.dump(annotation_ret, fh,  indent=4)


if __name__ == '__main__':
    main()
    main('val')
