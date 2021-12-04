# VTPACK
This repo is an official implementation for "[Dynamic Grained Encoder for Vision Transformers](https://openreview.net/pdf/6f44f2a1eb19252a3640dda8d564fef11b090246.pdf)" (NeurIPS2021) on PyTorch framework. 


## Installation
### Requirements
- Python >= 3.6
- PyTorch >= 1.8 and torchvision
- timm: 
	- `pip install timm`
- GCC >= 4.9

### Build from source
- `git clone https://github.com/StevenGrove/vtpack`
- `cd vtpack`
- `python setup.py build develop`

### Prepare data
Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Usage

### Training
```
# Running training procedure with specific GPU number
./tools/run_dist_launch.sh <GPU_NUM> <path_to_config> [optional arguments]

# Please refer to main.py for more optional arguments
```

### Inference
```
# Running inference procedure with specific GPU number and model path
./tools/run_dist_launch.sh <GPU_NUM> <path_to_config> --eval --resume <model_path> [optional arguments]

# Please refer to main.py for more optional arguments
```

### Image Classification on ImageNet *val* set

The following models are trained and evaluated with 256 * 256 input images. The budget for DGE is 0.5.
 Method | Acc1 | Acc5 (%) |  MAC<sub>avg</sub> | Project | Model
:--|:--:|:--:|:--:|---|---
 DeiT-Ti | 73.2 | 91.8 | 1.7G | [Link](configs/standard/deit_tiny_patch16_256_mm4.sh) | [GoogleDrive](https://drive.google.com/file/d/10G4nvjV87q_h5q3jaAkoFbYYYJ20LNBr/view?usp=sharing)
 DeiT-Ti + DGE | 73.2 | 91.7 | 1.1G | [Link](configs/dge/deit_dge_s124_b0_5_tiny_mm4.sh) | [GoogleDrive](https://drive.google.com/file/d/1Cie5Ylmf1_qTqim-tPls6upRvyjD8Sy_/view?usp=sharing)
 DeiT-S | 80.6 | 95.4 | 6.1G | [Link](configs/standard/deit_small_patch16_256_mm4.sh) | [GoogleDrive](https://drive.google.com/file/d/1N31RDWWIN4uR9qTETK_zqs7eYmdRzuYV/view?usp=sharing)
 DeiT-S + DGE | 80.1 | 95.0 | 3.5G | [Link](configs/dge/deit_dge_s124_b0_5_small_mm4.sh) | [GoogleDrive](https://drive.google.com/file/d/17tgI4j_ZiqA4gzLKB9miDKgQ3-k_yzFR/view?usp=sharing)

### More models are comming soon.

## Citation

Please cite the paper in your publications if it helps your research.

```
@inproceedings{song2021dynamic,
    title={Dynamic Grained Encoder for Vision Transformers},
    author={Song, Lin and Zhang, Songyang and Liu, Songtao and Li, Zeming and He, Xuming and Sun, Hongbin and Sun, Jian and Zheng, Nanning},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021}
}
```

Please cite this project in your publications if it helps your research.
```
@misc{vtpack,
    author = {Song, Lin},
    title = {VTPACK},
    howpublished = {\url{https://github.com/StevenGrove/vtpack}},
    year ={2021}
}
```
