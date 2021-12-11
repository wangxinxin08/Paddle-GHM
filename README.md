

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
| this                                                         | R-50-FPN | 12           | 37.4 | config                                                       | [baidu](https://pan.baidu.com/s/1UmUXq6UKZuQhvld9SXXGKg)[ahbr] | log                                                          |

The experiments are performed on coco 2017 dataset. The training and testing are both done at  [Baidu AIStudio](https://aistudio.baidu.com/aistudio/index). The model is trained on 4 V100 GPUs with 2 images per GPU. 

Please check out the config for more information on the model.

Please check out the train-log for more information on the loss during training. 

Data in the first two rows are directly taken from official and mmdet's github repo, their dataset should also be coco 2017 according to [this](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md). 

NOTICE: official and mmdet are based on Pytorch and MMDetection while this is based on Paddle and PaddleDetection.

## Usage

Requirements: 

python 3.7+

Paddle v2.2: follow [this](https://www.paddlepaddle.org.cn/install/quick) to install

PaddleDetection v2.3: follow [this](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/INSTALL.md) to install, follow [this](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/GETTING_STARTED.md) to get started. 



Train GHM on a single GPU:

```python
python tools/train.py -c configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml --eval
```

Train GHM on multiple GPUs:

```python
python -m paddle.distributed.launch --gpus 0,1,2,3 PaddleDetection/tools/train.py -c configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml --eval
```

Eval AP of GHM:

```python
python PaddleDetection/tools/eval.py -c configs/retinanet_ghm_r50_fpn_1x_coco_2x4GPU.yml -o weights=path_to_model_final.pdparams
```



## Acknowledgement 

I want to thank [Baidu AIStudio](https://aistudio.baidu.com/aistudio/index) for providing good amount of GPU computing power. 

As well as the following amazing open-source projects:

[GHM Official Code Release](https://github.com/libuyu/GHM_Detection)

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

[MMDetection](https://github.com/open-mmlab/mmdetection)

