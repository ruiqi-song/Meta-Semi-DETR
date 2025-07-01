# Meta Semi-DETR

This repo is the official implementation of paper ["Meta Semi-DETR: Semantic Prior Queries for Semi-Supervised Object Detection"](./assets/framework.png).

Meta Semi-DETR, a novel semi-supervised object detection framework that is built upon the use of large-scale vision language models to automatically generate textual descriptions from unlabeled images. These descriptions are encoded into semantic prior queries and injected into the DETR architecture to enhance the semantic representation of object queries and improve detection performance.

This repo is mostly built on top of [Semi-DETR](https://github.com/JCZ404/Semi-DETR). The Q-Adapter module is built on top of initial implementation in [LAVIS](https://github.com/salesforce/LAVIS).

# News

- **`[2025/6/25]`:** Initial training&evaluation code release.

# Methods

## Framework

<p align="center">
    <img src=assets/framework.png width="95%" style="display: inline-block; margin-right: 2%;" />
</p>

## Highlights

- This is the first work to explore how VLMs can be leveraged to distill and enhance semi-supervised object detection capabilities.
- We focus on introducing an effective strategy for injecting semantic information, conveniently obtained from unlabeled images via VLMs, into the SSOD pipeline.
- A semantic consistency regularization mechanism, which aligns the textual semantics derived from different augmented views, is proposed.
- Extensive experiments on the MS-COCO benchmark demonstrate that Meta SemiDETR achieves state-of-the-art performance.
- To verify its generalization and practicality, we build a pipeline for industrial-scale autonomous driving with prompt engineering, text generation, and semi-supervised training, evaluated on nuScenes.
- It requires only about 34 hours of computation on a single NVIDIA L20 for text annotation without any manual involvement, which show efficiency and promising potential for real-world deployment.
- We contribute scene-level textual descriptions for the nuScenes, which enhances its potential for advancing research in multimodal learning.

# Setup

<details>

<summary>Environment Installation</summary>

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

- Download bert-base-uncased model weight from [this link](https://huggingface.co/google-bert/bert-base-uncased) and put it under `weights` folder.

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
  	├── instances_val2017.json
  	├── captions_train2017.json
  	└── captions_val2017.json
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

## Training

- To train model on the **COCO partial labeled data** setting:

  ```shell script
  bash dist_train.sh configs/metadetr_ssod_dino_r50_coco_120k.py ${FOLD} ${PERCENT}
  ```

- To train model on the **NuScenes partial labeled data** setting:

  ```shell script
  bash dist_train.sh configs/metadetr_ssod_dino_r50_nusc_20k.py ${FOLD} ${PERCENT}
  ```

  For example, you can run the following scripts to train metasemi-detr on 10% labeled COCO data with 8 GPUs on 1th split:

  ```shell script
  bash dist_train.sh configs/metadetr_ssod_dino_r50_coco_120k.py 1 10
  ```

## Evaluation

- To eval model:

  ```shell script
  python tools/test.py ${CONFIG_FILE_PATH} ${CHECKPOINT_PATH} --eval bbox
  ```

- **COCO Evaluation metrics:**

  Detailed evaluation metrics for the COCO-Partial setting:
  | Data Setting | mAP mAP_50 mAP_75 mAP_s mAP_m mAP_l | Per-class AP | Checkpoint |
  | ------- | ------- | ------------------------ | ------- |
  | 1% Data |0.350 0.510 0.372 0.192 0.379 0.480 | 🔽More | [Google Drive]()|
  | 5% Data |0.422 0.586 0.453 0.237 0.460 0.574 | 🔽More | [Google Drive]()|
  | 10% Data |0.451 0.618 0.486 0.270 0.490 0.599| 🔽More | [Google Drive]()|

  Comparing Mate Semi-DETR with latest SOTA methods on COCO-Partial setting:
    <p align="left">
        <img src=assets/eval_coco.png width="95%" style="display: inline-block; margin-right: 2%;" />
    </p>

    <details open>

    <summary>Per-class AP details with 1% data setting</summary>

  ```ASCII
  +---------------+-------+--------------+-------+----------------+-------+
  | category      | AP    | category     | AP    | category       | AP    |
  +---------------+-------+--------------+-------+----------------+-------+
  | person        | 0.495 | bicycle      | 0.257 | car            | 0.378 |
  | motorcycle    | 0.336 | airplane     | 0.651 | bus            | 0.594 |
  | train         | 0.598 | truck        | 0.275 | boat           | 0.178 |
  | traffic light | 0.204 | fire hydrant | 0.607 | stop sign      | 0.541 |
  | parking meter | 0.377 | bench        | 0.191 | bird           | 0.317 |
  | cat           | 0.714 | dog          | 0.624 | horse          | 0.538 |
  | sheep         | 0.467 | cow          | 0.555 | elephant       | 0.590 |
  | bear          | 0.732 | zebra        | 0.650 | giraffe        | 0.655 |
  | backpack      | 0.068 | umbrella     | 0.306 | handbag        | 0.075 |
  | tie           | 0.288 | suitcase     | 0.287 | frisbee        | 0.613 |
  | skis          | 0.126 | snowboard    | 0.246 | sports ball    | 0.406 |
  | kite          | 0.372 | baseball bat | 0.232 | baseball glove | 0.342 |
  | skateboard    | 0.464 | surfboard    | 0.335 | tennis racket  | 0.404 |
  | bottle        | 0.319 | wine glass   | 0.284 | cup            | 0.346 |
  | fork          | 0.240 | knife        | 0.092 | spoon          | 0.076 |
  | bowl          | 0.359 | banana       | 0.159 | apple          | 0.088 |
  | sandwich      | 0.260 | orange       | 0.242 | broccoli       | 0.199 |
  | carrot        | 0.095 | hot dog      | 0.254 | pizza          | 0.517 |
  | donut         | 0.377 | cake         | 0.317 | chair          | 0.215 |
  | couch         | 0.391 | potted plant | 0.129 | bed            | 0.421 |
  | dining table  | 0.236 | toilet       | 0.575 | tv             | 0.512 |
  | laptop        | 0.525 | mouse        | 0.568 | remote         | 0.160 |
  | keyboard      | 0.442 | cell phone   | 0.235 | microwave      | 0.488 |
  | oven          | 0.294 | toaster      | 0.148 | sink           | 0.295 |
  | refrigerator  | 0.522 | book         | 0.078 | clock          | 0.462 |
  | vase          | 0.313 | scissors     | 0.169 | teddy bear     | 0.404 |
  | hair drier    | 0.001 | toothbrush   | 0.091 | None           | None  |
  +---------------+-------+--------------+-------+----------------+-------+
  ```

    </details>

    <details open>

    <summary>Per-class AP details with 5% data setting</summary>

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

  <details open>

  <summary>Per-class AP details with 10% data setting</summary>

  ```ASCII
  +---------------+-------+--------------+-------+----------------+-------+
  | category      | AP    | category     | AP    | category       | AP    |
  +---------------+-------+--------------+-------+----------------+-------+
  | person        | 0.578 | bicycle      | 0.329 | car            | 0.469 |
  | motorcycle    | 0.470 | airplane     | 0.721 | bus            | 0.698 |
  | train         | 0.683 | truck        | 0.385 | boat           | 0.273 |
  | traffic light | 0.290 | fire hydrant | 0.699 | stop sign      | 0.646 |
  | parking meter | 0.516 | bench        | 0.267 | bird           | 0.412 |
  | cat           | 0.762 | dog          | 0.702 | horse          | 0.649 |
  | sheep         | 0.572 | cow          | 0.626 | elephant       | 0.717 |
  | bear          | 0.784 | zebra        | 0.727 | giraffe        | 0.736 |
  | backpack      | 0.134 | umbrella     | 0.401 | handbag        | 0.142 |
  | tie           | 0.366 | suitcase     | 0.461 | frisbee        | 0.674 |
  | skis          | 0.299 | snowboard    | 0.421 | sports ball    | 0.475 |
  | kite          | 0.503 | baseball bat | 0.326 | baseball glove | 0.416 |
  | skateboard    | 0.566 | surfboard    | 0.439 | tennis racket  | 0.495 |
  | bottle        | 0.409 | wine glass   | 0.400 | cup            | 0.451 |
  | fork          | 0.417 | knife        | 0.229 | spoon          | 0.181 |
  | bowl          | 0.474 | banana       | 0.271 | apple          | 0.189 |
  | sandwich      | 0.398 | orange       | 0.312 | broccoli       | 0.224 |
  | carrot        | 0.217 | hot dog      | 0.412 | pizza          | 0.553 |
  | donut         | 0.516 | cake         | 0.413 | chair          | 0.301 |
  | couch         | 0.466 | potted plant | 0.261 | bed            | 0.520 |
  | dining table  | 0.311 | toilet       | 0.647 | tv             | 0.608 |
  | laptop        | 0.653 | mouse        | 0.635 | remote         | 0.361 |
  | keyboard      | 0.557 | cell phone   | 0.383 | microwave      | 0.624 |
  | oven          | 0.384 | toaster      | 0.126 | sink           | 0.391 |
  | refrigerator  | 0.669 | book         | 0.154 | clock          | 0.538 |
  | vase          | 0.402 | scissors     | 0.327 | teddy bear     | 0.503 |
  | hair drier    | 0.011 | toothbrush   | 0.357 | None           | None  |
  +---------------+-------+--------------+-------+----------------+-------+
  ```

    </details>

- **NuScenes Evaluation metrics:**

  Detailed evaluation metrics for the Nuscenes-Partial setting:
  | Data Setting | mAP mAP_50 mAP_75 mAP_s mAP_m mAP_l | Per-class AP | Checkpoint |
  | ------- | ------- | ------------------------ | ------- |
  | 1% Data | 0.260 0.485 0.253 0.040 0.195 0.381| 🔽More | [Google Drive]()|
  | 5% Data | 0.320 0.584 0.316 0.049 0.257 0.445| 🔽More |[Google Drive]() |
  | 10% Data |0.327 0.589 0.330 0.060 0.272 0.450 | 🔽More |[Google Drive]() |

  Comparing Mate Semi-DETR with latest SOTA methods on Nuscenes-Partial setting:
  <p align="left">
      <img src=assets/eval_nusc.png width="95%" style="display: inline-block; margin-right: 2%;" />
  </p>

  <details open>
  <summary>Per-class AP details with 1% data setting</summary>

  ```ASCII
  +------------+-------+----------------------+-------+--------------+-------+
  | category   | AP    | category             | AP    | category     | AP    |
  +------------+-------+----------------------+-------+--------------+-------+
  | car        | 0.502 | truck                | 0.326 | bus          | 0.518 |
  | trailer    | 0.118 | construction_vehicle | 0.005 | pedestrian   | 0.278 |
  | motorcycle | 0.203 | bicycle              | 0.185 | traffic_cone | 0.219 |
  | barrier    | 0.248 | None                 | None  | None         | None  |
  +------------+-------+----------------------+-------+--------------+-------+
  ```

  </details>

  <details open>

  <summary>Per-class AP details with 5% data setting</summary>

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
    <details open>

    <summary>Per-class AP details with 10% data setting</summary>

  ```ASCII
  +------------+-------+----------------------+-------+--------------+-------+
  | category   | AP    | category             | AP    | category     | AP    |
  +------------+-------+----------------------+-------+--------------+-------+
  | car        | 0.546 | truck                | 0.376 | bus          | 0.566 |
  | trailer    | 0.172 | construction_vehicle | 0.070 | pedestrian   | 0.322 |
  | motorcycle | 0.272 | bicycle              | 0.269 | traffic_cone | 0.309 |
  | barrier    | 0.370 | None                 | None  | None         | None  |
  +------------+-------+----------------------+-------+--------------+-------+
  ```

    </details>

# Datainfo

- Scene-level textual descriptions for nuScenes:

| NuScenes Image                             | Caption generated by Ovis1.6-Gemma2-9B-gptq-int4 using manually designed prompt                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src=assets/image_1.png width=1200  /> | The image depicts an urban street scene with several vehicles, pedestrians, and traffic signs. There are at least six vehicles visible, including cars and an SUV, primarily on the road. Pedestrians are seen on the sidewalk, near the crosswalk. Traffic signs are present at intersections, including traffic lights and street signs. The scene is set against a backdrop of construction and modern buildings, indicating a busy city environment. |
| <img src=assets/image_3.png width=1000  /> | The image depicts an urban street scene with several vehicles parked on the left side, including a black SUV and a white sedan. A pedestrian in a wheelchair is visible on the sidewalk. Traffic signs are present, including a stop sign near the center. The scene is characterized by older buildings and a modern skyscraper in the background.                                                                                                      |
| <img src=assets/image_2.png width=1000  /> | The image depicts a busy urban road with a line of vehicles, including a silver car in the foreground, a white truck, and several other cars, all moving slowly or stopped. A black SUV is visible on the right. There are no pedestrians or traffic signs in the image. The scene is set against a backdrop of overcast skies and urban buildings.                                                                                                      |

# Acknowledgement

Many thanks to these excellent open source projects:

- [Semi-DETR](https://github.com/JCZ404/Semi-DETR)
- [LAVIS](https://github.com/salesforce/LAVIS)
- [DINO](https://github.com/IDEA-Research/DINO)
- [Ovis](https://github.com/AIDC-AI/Ovis)

# References
