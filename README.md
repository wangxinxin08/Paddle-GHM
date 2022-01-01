

## Introduction

This project is trying to reproduce model GHM, which is proposed in paper [Gradient Harmonized Single-stage Detector](https://arxiv.org/abs/1811.05181). There are two Pytorch-based implementations under consideration:

[Official Implementation](https://github.com/libuyu/GHM_Detection)

[MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.16.0/configs/ghm)

But since the official implementation is based on MMDetection, they can be viewed as one. 

GHM is essentially RetinaNet with it's classification loss and regression loss replaced by GHMC and GHMR. GHMC and GHMR are two novel losses proposed in the paper, where the loss weight of a loss is associated with distribution of the gradient. The purpose of GHM losses is the same as Focal Loss, they are all trying to balance loss between hard and easy samples. GHM is better than Focal Loss in that it also down-weights loss from potential noisy samples, according to the paper. Check out [the paper](https://arxiv.org/abs/1811.05181) for more details.

## Implementation

The implementation is based on PaddleDetection. PaddleDetection has already implemented the backbone, the neck and the training utilities, so I directly used them. What I have added is the implementation of RetinaNet and GHM losses.

The design of RetinaNet follows PaddleDetection's GFL, the training process follows MMDetection's RetinaNet implementation.

The GHM losses follow official/MMDetection. 

All the code is located at: `PaddleDetection/ppdet/ghm`.

## Results

| source                                                       | backbone | total epochs | AP   | config                                                       | model                                                        | train-log                                                    |
| ------------------------------------------------------------ | -------- | ------------ | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [official](https://github.com/libuyu/GHM_Detection)          | R-50-FPN | 12           | 37.0 | NA                                                           | NA                                                           | NA                                                           |
| [mmdet](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm) | R-50-FPN | 12           | 37.0 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth) | [log](https://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130_004213.log.json) |
| this                                                         | R-50-FPN | 12           | 37.4 | [config](https://github.com/thisisi3/Paddle-GHM/blob/main/configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml) | [baidu](https://pan.baidu.com/s/1UmUXq6UKZuQhvld9SXXGKg)[ahbr] | [log](https://github.com/thisisi3/Paddle-GHM/blob/main/workerlog.0) |

The experiments are performed on coco 2017 dataset. The training and testing are both done at Baidu AIStudio. The model is trained on 4 V100 GPUs with 2 images per GPU. 

Please check out the [config](https://github.com/thisisi3/Paddle-GHM/blob/main/configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml) for more information on the model.

Please check out the [train-log](https://github.com/thisisi3/Paddle-GHM/blob/main/workerlog.0) for more information on the loss during training. 

Data in the first two rows are directly taken from official and mmdet's github repo, their dataset should also be coco 2017 according to [this](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md). 

NOTICE: official and mmdet are based on Pytorch and MMDetection while this is based on Paddle and PaddleDetection.

## Usage

Requirements: 

- python 3.7+

- Paddle v2.2: follow [this](https://www.paddlepaddle.org.cn/install/quick) to install


Clone this repo and install:

```shell
git clone https://github.com/thisisi3/Paddle-GHM.git
pip install -e Paddle-GHM/PaddleDetection -v
```

Follow [this](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/INSTALL.md) for more details on installation of PaddleDetection, follow [this](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/GETTING_STARTED.md) to learn how to use PaddleDetection. 

Data preparation:

```shell
cd Paddle-GHM

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

mkdir dataset
mkdir dataset/coco

unzip annotations_trainval2017.zip -d dataset/coco
unzip train2017.zip -d dataset/coco
unzip val2017.zip -d dataset/coco
```

You can also go to [aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/7122) to download coco 2017 if official download is slow.

Train GHM on a single GPU:

```shell
python PaddleDetection/tools/train.py -c configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml --eval
```

Train GHM on multiple GPUs:

```shell
python -m paddle.distributed.launch --gpus 0,1,2,3 PaddleDetection/tools/train.py -c configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml --eval
```

Eval AP of GHM:

```shell
python PaddleDetection/tools/eval.py -c configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml -o weights=path_to_model_final.pdparams
```

Quick demo:

PaddleDetection comes with a inference script that allows us to visualize detection results. I provide an example image at `demo/` and by running the following command we can visualize detection results of GHM on the image:

```shell
python PaddleDetection/tools/infer.py -c configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml -o weights=path_to_model_final.pdparams --infer_img demo/000000371552.jpg --output_dir demo/out/ --draw_threshold 0.6
```

The example image:

![](https://github.com/thisisi3/Paddle-GHM/blob/main/demo/000000371552.jpg?raw=true)

Add detection bboxes:

![](https://github.com/thisisi3/Paddle-GHM/blob/main/demo/out/000000371552.jpg?raw=true)



## Acknowledgement 

I want to thank [Baidu AIStudio](https://aistudio.baidu.com/aistudio/index) for providing good amount of GPU computing power. 

As well as the following amazing open-source projects:

[GHM Official Code Release](https://github.com/libuyu/GHM_Detection)

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

[MMDetection](https://github.com/open-mmlab/mmdetection)

