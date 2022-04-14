<!--- # STME:A Spatiotemporal and Motion Information Extraction Network for Action Recognition -->
## Prerequisites
The code is built with following libraries:<br>
* Python >= 3.6
* [Pytorch](https://pytorch.org/) >= 1.6
* [Torchvision](https://github.com/pytorch/vision)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [TensorboardX](https://github.com/lanpa/tensorboardX)
* [decord](https://github.com/dmlc/decord)
* [tqdm](https://github.com/tqdm/tqdm)
* [termcolor](https://github.com/ikalnytskyi/termcolor)
* [ffmpeg](https://www.ffmpeg.org/)
## Pretrained models
Here we provide some [pretrained models](https://drive.google.com/drive/folders/1eN-1VPw7Kb9KKDImFjEGlsouFimPQXG-) of STME.<br>
(To be updated)
## Testing and training
### Testing
For example, to test the downloaded [pretrained models](https://drive.google.com/drive/folders/1eN-1VPw7Kb9KKDImFjEGlsouFimPQXG-), you can run as following steps:
#### Efficient setting (center corp and 1 clip)
  ```
  CUDA_VISIBLE_DEVICES=0 python test_models_center_crop.py something \ 
  --archs=resnet50 --weights=your_checkpoint_path/STME_something_RGB_resnet50_avg_segment8_e60.pth.tar \ 
  --test_segments=8  --test_crops=1 --batch_size=16 --gpus=0 --output_dir=your_pkl_path --workers=4 --clip_index=0
  ```
  ```
  python pkl_to_results.py --num_clips=1 --test_crops=1 --output_dir=your_pkl_path
  ```
#### Accurate setting (full resolution and 10 clips)
  ```
  CUDA_VISIBLE_DEVICES=0 python test_models_three_crops.py spmething \ 
  --archs=resnet50 --weights=your_checkpoint_path \ 
  --test_segments=8  --test_crops=3 --batch_size=16 --gpus=0 \
  --full_res --output_dir=your_pkl_path --workers=4 --clip_index=0
  ```
  you should test this scrips for 10 times and modify clip_index from 0 to 9.
  ```
  python pkl_to_results.py --num_clips=10 --test_crops=3 --output_dir=your_pkl_path
  ```  
### Training
To train STME-ResNet on Something-Something dataset, you can run:
  ```
  python -m torch.distributed.launch --nproc_per_node=4 \
  main.py something RGB --arch=resnet50 --num_segments=8 --gd 20 \
  --lr 0.01 --lr_scheduler step --lr_steps 30 45 55 --epochs 60 --batch-size=16 \
  --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 --workers=4 --npb
  ```
### Acknowledgment
Our codes are built based on previous repos [TSN](https://github.com/yjxiong/temporal-segment-networks), [TSM](https://github.com/mit-han-lab/temporal-shift-module), and [TDN](https://github.com/MCG-NJU/TDN).
