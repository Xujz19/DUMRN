# Deep unfolding multi-scale regularizer network for image denoising

Training and testing code for our paper "Deep unfolding multi-scale regularizer network for image denoising"

## Requirements

- Pytorch == 1.1.0

## Training

### Prepare training data

Download 800 DIV2K training images from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

### Gray image denoising

cd to './code/Gray' and run the following script

```
CUDA_VISIBLE_DEVICES=0 python main_train.py --train_data path_of_training_data --n_colors 1 --sigma noise_level
```

### Color image denoising

cd to './code/RGB' and run the following script

```
CUDA_VISIBLE_DEVICES=0 python main_train.py --train_data path_of_training_data --n_colors 3 --sigma noise_level
```

## Testing

Download pretrained models for [gray](https://drive.google.com/file/d/1w6985fnUasKoEOEyBvv64gFGXHZnyK3J/view?usp=sharing) and color image denoising and place them to './Gray/experiment/models' and './RGB/experiment/models' respectively.

Download test datasets from [Google Drive](https://drive.google.com/file/d/1BIuUOUUOjjlihkV-Bcpu41a-42X-_DnR/view?usp=sharing)

### Gray image denoising

cd to './code/Gray' and run the following script

```
CUDA_VISIBLE_DEVICES=0 python main_test.py --test_data path_of_training_data --n_colors 1 --dataset_name test_dataset_name --sigma noise_level --save_results True
```

### Color image denoising

cd to './code/RGB' and run the following script

```
CUDA_VISIBLE_DEVICES=0 python main_test.py --test_data path_of_training_data --n_colors 3 --dataset_name test_dataset_name --sigma noise_level --save_results True
```

Gray image denoising results can be downloaded from [here](https://drive.google.com/file/d/1vLvY8iuDB2VtD5mfpgO2-R0KSW-yXrFy/view?usp=sharing).
Color image denoising results can be downloaded from [here](https://drive.google.com/file/d/1nY0DcXSxy2AGY2BqGJk-aLdB8xmAJ4q4/view?usp=sharing).


## Citation
```
@article{xu2022deep,
  title={Deep Unfolding Multi-scale Regularizer Network for Image Denoising},
  author={Xu, Jingzhao and Yuan, Mengke and Yan, Dong-Ming and Wu, Tieru},
  journal={Computational Visual Media},
  year={2022}
}
```