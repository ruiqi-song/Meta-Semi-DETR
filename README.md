# Meta Semi-DETR

This repo is the official implementation of paper ["Meta Semi-DETR: Semantic Prior Queries for Semi-Supervised Object Detection"](./assets/framework.png).

Meta Semi-DETR, a novel semi-supervised object detection framework that is built upon the use of large-scale vision language models to automatically generate textual descriptions from unlabeled images. These descriptions are encoded into semantic prior queries and injected into the DETR architecture to enhance the semantic representation of object queries and improve detection performance.

This repo is mostly built on top of [Semi-DETR](https://github.com/JCZ404/Semi-DETR). The Q-Adapter module is built on top of initial implementation in [LAVIS](https://github.com/salesforce/LAVIS).

# News

- **`[2025/6/25]`:** Initial training&evaluation code release.

# Methods

<p align="center">
    <img src=./assets/framework.png width="95%" style="display: inline-block; margin-right: 2%;" />
</p>

# Setup

<details>

<summary>Installation</summary>

```bash
# Create virtual environment
conda create -n metasemidetr python=3.8 -y
conda activate metasemidetr

# Install PyTorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

# Install mmcv-full
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.1/index.html

# Install lavis
pip install spacy==3.7.5
pip install salesforce-lavis

# Install mmdetection
pip install -v -e ./3rdparty/mmdetection

# Install CUDA ops for deformable attention
cd detr_od/models/utils/ops
python setup.py build install

# Install detr_ssod
pip install -v -e .

pip install prettytable
pip install yapf==0.40.1
pip install nuscenes-devkit
```

</details>

<details>
<summary>Data Preparation</summary>

1. **COCO**

- Download the [COCO 2017](https://cocodataset.org/) dataset and organize them as following:

  ```bash
  dataset/coco2017/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
  ```

- Execute the following command to generate data set splits, or download the split files(refer to [Semi-DETR](https://github.dev/JCZ404/Semi-DETR)) from [here](https://drive.google.com/file/d/1Hq98YEU-WQXkZ6nR3t_OxuTNJKp5) and put them under `dataset/coco2017/annotations/semi_supervised/` folder:

  ```shell script
  bash tools/dataset/prepare_coco_data.sh conduct
  ```

- Execute the following command to export caption label from `captions_train2017.json`/`captions_val2017.json`, label files will be generated under `dataset/coco2017/captions_coco/` folder:

  ```shell script
  python tools/dataset/prepare_coco_cap.py
  ```

2. **NuScenes**

- Download the [NuScenes](https://www.nuscenes.org/nuscenes) dataset
- Execute the following command to export 2D annotations with coco format, and organize them as following:

  ```bash
  dataset/nuscenes_coco/
  ├── nuscenes/
  ├── images/
  └── annotations/
  	├── train.json
  	└── val.json

  python tools/dataset/export_nusc2coco.py
  ```

- Download the split files from [here](.) and put them under `dataset/nuscenes_coco/annotations/semi_supervised/` folder.
- Download the caption label files from [here](.) and put them under `dataset/nuscenes_coco/captions_ovis/` folder.

</details>

# Training&Evaluation

- Download bert-base-uncased model weight from [this link](https://huggingface.co/google-bert/bert-base-uncased) and put it under `weights/` folder.

## Training

- To train model on the **COCO partial labeled data** setting:

  ```shell script
  bash dist_train.sh configs/metadetr_ssod_dino_r50_coco_120k.py ${FOLD} ${PERCENT}
  ```

- To train model on the **NuScenes partial labeled data** setting:

  ```shell script
  bash dist_train.sh configs/metadetr_ssod_dino_r50_nusc_20k.py ${FOLD} ${PERCENT}
  ```

- For example, you can run the following scripts to train metasemi-detr on 10% labeled COCO data with 8 GPUs on 1th split:

  ```shell script
  bash dist_train.sh configs/metadetr_ssod_dino_r50_coco_120k.py 1 10
  ```

## Evaluation

- To eval model:

  ```shell script
  python tools/test.py ${CONFIG_FILE_PATH} ${CHECKPOINT_PATH} --eval bbox
  ```

- **COCO**
  | Data Setting | mAP mAP_50 mAP_75 mAP_s mAP_m mAP_l | Details | Checkpoint |
  | ------- | ------- | ------------------------ | ------- |
  | 1% Data |0.351 0.512 0.373 0.192 0.378 0.481 | 🔽More | Google Drive|
  | 5% Data |0.422 0.586 0.453 0.237 0.460 0.574 | 🔽More | [Google Drive]()|
  | 10% Data | 0.451 0.616 0.488 0.269 0.493 0.602| 🔽More | Google Drive|

    <details>

    <summary>1% Data Details</summary>

  ```ASCII
  +---------------+-------+--------------+-------+----------------+-------+
  | category      | AP    | category     | AP    | category       | AP    |
  +---------------+-------+--------------+-------+----------------+-------+
  | person        | 0.499 | bicycle      | 0.258 | car            | 0.377 |
  | motorcycle    | 0.324 | airplane     | 0.637 | bus            | 0.593 |
  | train         | 0.603 | truck        | 0.276 | boat           | 0.185 |
  | traffic light | 0.208 | fire hydrant | 0.610 | stop sign      | 0.525 |
  | parking meter | 0.369 | bench        | 0.194 | bird           | 0.324 |
  | cat           | 0.711 | dog          | 0.632 | horse          | 0.541 |
  | sheep         | 0.471 | cow          | 0.561 | elephant       | 0.578 |
  | bear          | 0.737 | zebra        | 0.650 | giraffe        | 0.658 |
  | backpack      | 0.073 | umbrella     | 0.309 | handbag        | 0.072 |
  | tie           | 0.284 | suitcase     | 0.292 | frisbee        | 0.619 |
  | skis          | 0.130 | snowboard    | 0.258 | sports ball    | 0.403 |
  | kite          | 0.377 | baseball bat | 0.235 | baseball glove | 0.341 |
  | skateboard    | 0.469 | surfboard    | 0.342 | tennis racket  | 0.402 |
  | bottle        | 0.321 | wine glass   | 0.280 | cup            | 0.349 |
  | fork          | 0.240 | knife        | 0.097 | spoon          | 0.077 |
  | bowl          | 0.360 | banana       | 0.163 | apple          | 0.089 |
  | sandwich      | 0.251 | orange       | 0.239 | broccoli       | 0.197 |
  | carrot        | 0.101 | hot dog      | 0.247 | pizza          | 0.523 |
  | donut         | 0.394 | cake         | 0.314 | chair          | 0.220 |
  | couch         | 0.384 | potted plant | 0.131 | bed            | 0.426 |
  | dining table  | 0.233 | toilet       | 0.576 | tv             | 0.520 |
  | laptop        | 0.524 | mouse        | 0.563 | remote         | 0.162 |
  | keyboard      | 0.456 | cell phone   | 0.240 | microwave      | 0.481 |
  | oven          | 0.286 | toaster      | 0.132 | sink           | 0.300 |
  | refrigerator  | 0.532 | book         | 0.080 | clock          | 0.461 |
  | vase          | 0.310 | scissors     | 0.220 | teddy bear     | 0.406 |
  | hair drier    | 0.001 | toothbrush   | 0.066 | None           | None  |
  +---------------+-------+--------------+-------+----------------+-------+
  ```

    </details>

    <details>

    <summary>5% Data Details</summary>

  ```ASCII
  +---------------+-------+--------------+-------+----------------+-------+
  | category      | AP    | category     | AP    | category       | AP    |
  +---------------+-------+--------------+-------+----------------+-------+
  | person        | 0.555 | bicycle      | 0.299 | car            | 0.434 |
  | motorcycle    | 0.437 | airplane     | 0.698 | bus            | 0.683 |
  | train         | 0.672 | truck        | 0.373 | boat           | 0.238 |
  | traffic light | 0.262 | fire hydrant | 0.701 | stop sign      | 0.621 |
  | parking meter | 0.510 | bench        | 0.247 | bird           | 0.389 |
  | cat           | 0.754 | dog          | 0.673 | horse          | 0.603 |
  | sheep         | 0.547 | cow          | 0.595 | elephant       | 0.678 |
  | bear          | 0.775 | zebra        | 0.696 | giraffe        | 0.724 |
  | backpack      | 0.121 | umbrella     | 0.370 | handbag        | 0.112 |
  | tie           | 0.334 | suitcase     | 0.395 | frisbee        | 0.674 |
  | skis          | 0.238 | snowboard    | 0.375 | sports ball    | 0.445 |
  | kite          | 0.447 | baseball bat | 0.316 | baseball glove | 0.373 |
  | skateboard    | 0.541 | surfboard    | 0.400 | tennis racket  | 0.475 |
  | bottle        | 0.379 | wine glass   | 0.371 | cup            | 0.416 |
  | fork          | 0.346 | knife        | 0.194 | spoon          | 0.153 |
  | bowl          | 0.412 | banana       | 0.229 | apple          | 0.117 |
  | sandwich      | 0.388 | orange       | 0.308 | broccoli       | 0.203 |
  | carrot        | 0.171 | hot dog      | 0.393 | pizza          | 0.539 |
  | donut         | 0.497 | cake         | 0.391 | chair          | 0.268 |
  | couch         | 0.435 | potted plant | 0.233 | bed            | 0.471 |
  | dining table  | 0.284 | toilet       | 0.637 | tv             | 0.577 |
  | laptop        | 0.604 | mouse        | 0.621 | remote         | 0.297 |
  | keyboard      | 0.518 | cell phone   | 0.342 | microwave      | 0.556 |
  | oven          | 0.359 | toaster      | 0.265 | sink           | 0.380 |
  | refrigerator  | 0.597 | book         | 0.128 | clock          | 0.514 |
  | vase          | 0.397 | scissors     | 0.256 | teddy bear     | 0.512 |
  | hair drier    | 0.018 | toothbrush   | 0.234 | None           | None  |
  +---------------+-------+--------------+-------+----------------+-------+
  ```

  </details>

  <details>

  <summary>10% Data Details</summary>

  ```ASCII
  +---------------+-------+--------------+-------+----------------+-------+
  | category      | AP    | category     | AP    | category       | AP    |
  +---------------+-------+--------------+-------+----------------+-------+
  | person        | 0.579 | bicycle      | 0.327 | car            | 0.463 |
  | motorcycle    | 0.459 | airplane     | 0.715 | bus            | 0.699 |
  | train         | 0.684 | truck        | 0.382 | boat           | 0.278 |
  | traffic light | 0.283 | fire hydrant | 0.712 | stop sign      | 0.655 |
  | parking meter | 0.501 | bench        | 0.255 | bird           | 0.408 |
  | cat           | 0.767 | dog          | 0.704 | horse          | 0.652 |
  | sheep         | 0.569 | cow          | 0.627 | elephant       | 0.721 |
  | bear          | 0.812 | zebra        | 0.725 | giraffe        | 0.746 |
  | backpack      | 0.141 | umbrella     | 0.417 | handbag        | 0.142 |
  | tie           | 0.369 | suitcase     | 0.465 | frisbee        | 0.673 |
  | skis          | 0.284 | snowboard    | 0.443 | sports ball    | 0.488 |
  | kite          | 0.486 | baseball bat | 0.333 | baseball glove | 0.406 |
  | skateboard    | 0.584 | surfboard    | 0.442 | tennis racket  | 0.493 |
  | bottle        | 0.410 | wine glass   | 0.398 | cup            | 0.444 |
  | fork          | 0.406 | knife        | 0.222 | spoon          | 0.189 |
  | bowl          | 0.478 | banana       | 0.269 | apple          | 0.167 |
  | sandwich      | 0.412 | orange       | 0.282 | broccoli       | 0.222 |
  | carrot        | 0.215 | hot dog      | 0.420 | pizza          | 0.556 |
  | donut         | 0.545 | cake         | 0.402 | chair          | 0.309 |
  | couch         | 0.481 | potted plant | 0.274 | bed            | 0.508 |
  | dining table  | 0.315 | toilet       | 0.683 | tv             | 0.611 |
  | laptop        | 0.640 | mouse        | 0.639 | remote         | 0.343 |
  | keyboard      | 0.559 | cell phone   | 0.401 | microwave      | 0.633 |
  | oven          | 0.379 | toaster      | 0.080 | sink           | 0.400 |
  | refrigerator  | 0.670 | book         | 0.142 | clock          | 0.538 |
  | vase          | 0.394 | scissors     | 0.330 | teddy bear     | 0.514 |
  | hair drier    | 0.015 | toothbrush   | 0.329 | None           | None  |
  +---------------+-------+--------------+-------+----------------+-------+
  ```

    </details>

  Comparing Mate Semi-DETR with latest SOTA methods on COCO-Partial setting:
  <p align="left">
      <img src=./assets/eval_coco.png width="50%" style="display: inline-block; margin-right: 2%;" />
  </p>

- **NuScenes**
  | Data Setting | mAP mAP_50 mAP_75 mAP_s mAP_m mAP_l | Details | Checkpoint |
  | ------- | ------- | ------------------------ | ------- |
  | 1% Data | 0.262 0.509 0.242 0.032 0.203 0.375| 🔽More | Google Drive|
  | 5% Data | 0.320 0.584 0.316 0.049 0.257 0.445| 🔽More |[Google Drive]() |
  | 10% Data |0.329 0.591 0.330 0.062 0.270 0.453 | 🔽More |Google Drive |

  <details>
  <summary>1% Data Details</summary>

  ```ASCII
  +------------+-------+----------------------+-------+--------------+-------+
  | category   | AP    | category             | AP    | category     | AP    |
  +------------+-------+----------------------+-------+--------------+-------+
  | car        | 0.474 | truck                | 0.273 | bus          | 0.516 |
  | trailer    | 0.146 | construction_vehicle | 0.067 | pedestrian   | 0.258 |
  | motorcycle | 0.183 | bicycle              | 0.156 | traffic_cone | 0.253 |
  | barrier    | 0.290 | None                 | None  | None         | None  |
  +------------+-------+----------------------+-------+--------------+-------+
  ```

  </details>

    <details>

  <summary>5% Data Details</summary>

  ```ASCII
  +------------+-------+----------------------+-------+--------------+-------+
  | category   | AP    | category             | AP    | category     | AP    |
  +------------+-------+----------------------+-------+--------------+-------+
  | car        | 0.529 | truck                | 0.352 | bus          | 0.568 |
  | trailer    | 0.179 | construction_vehicle | 0.084 | pedestrian   | 0.306 |
  | motorcycle | 0.264 | bicycle              | 0.257 | traffic_cone | 0.306 |
  | barrier    | 0.356 | None                 | None  | None         | None  |
  +------------+-------+----------------------+-------+--------------+-------+
  ```

    </details>
    <details>

    <summary>10% Data Details</summary>

  ```ASCII
  +------------+-------+----------------------+-------+--------------+-------+
  | category   | AP    | category             | AP    | category     | AP    |
  +------------+-------+----------------------+-------+--------------+-------+
  | car        | 0.550 | truck                | 0.377 | bus          | 0.571 |
  | trailer    | 0.184 | construction_vehicle | 0.071 | pedestrian   | 0.320 |
  | motorcycle | 0.259 | bicycle              | 0.277 | traffic_cone | 0.312 |
  | barrier    | 0.368 | None                 | None  | None         | None  |
  +------------+-------+----------------------+-------+--------------+-------+
  ```

    </details>

  Comparing Mate Semi-DETR with latest SOTA methods on Nuscenes-Partial setting.
  <p align="left">
      <img src=./assets/eval_nusc.png width="50%" style="display: inline-block; margin-right: 2%;" />
  </p>

# Acknowledgement

Many thanks to these excellent open source projects:

- [Semi-DETR](https://github.com/JCZ404/Semi-DETR)
- [LAVIS](https://github.com/salesforce/LAVIS)
- [DINO](https://github.com/IDEA-Research/DINO)
- [Ovis](https://github.com/AIDC-AI/Ovis)

# References
