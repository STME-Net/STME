# STME:A Spatiotemporal and Motion Information Extraction Network for Action Recognition
## Overview
![2021-11-07_135150](https://user-images.githubusercontent.com/93808130/140634273-6155bacc-3186-48fa-9fda-7e27138643ab.jpg) <br>
We release our codes of STME. Our codes are built based on previos repos [TSN](https://github.com/yjxiong/temporal-segment-networks), [TSM](https://github.com/mit-han-lab/temporal-shift-module) and [TEA](https://github.com/Phoenix1327/tea-action-recognition). The core codes about our modules are in `ops/`.<br>
* [Prerequisites](#prerequisites)<br>
* [Data preparation](#data-preparation)<br>
* [Pretrained models](#pretrained-models)<br>
* [Testing and training](#testing-and-training)<br>
## Prerequisites
The code is built with following libraries:<br>
* Python >= 3.6
* [Pytorch](https://pytorch.org/) >= 1.6
* [Torchvision](https://github.com/pytorch/vision)
* [sciki-learn](https://github.com/scikit-learn/scikit-learn)
* [TensorboardX](https://github.com/lanpa/tensorboardX)
* [decord](https://github.com/dmlc/decord)
* [tqdm](https://github.com/tqdm/tqdm)
* [termcolor](https://github.com/ikalnytskyi/termcolor)
* [ffmpeg](https://www.ffmpeg.org/)
## Data preparation
We have successfully trained on [Jester](https://20bn.com/datasets/jester), [Something-Something V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2) datasets. For detailed data pre-processing, please refer [TSM](https://github.com/mit-han-lab/temporal-shift-module).<br>
Then, please refer [TSM/tools](https://github.com/mit-han-lab/temporal-shift-module/tree/master/tools) to generate data annotations.<br>
Finally, you should add the absolute path of your data annotations into `ops/dataset_configs.py`.

## Pretrained models

## Testing and training

