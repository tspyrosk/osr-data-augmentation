# osr-data-augmentation
Enhancing Robustness to Novel Visual Defects through StyleGAN Latent Space Navigation: A Manufacturing Use Case

### Overview

The main building blocks of the proect are the three notebooks representing the three steps in the data augmentation process:

sefa-shaver-generate-data.ipynb --> sefa-shaver-filter-data.ipynb --> sefa-shaver-osr-exps.ipynb (open-set recognition and evaluation)

The three scripts should be run in sequence and the output of each script should be placed in the folder expected by the next as described in the `Inputs` section.

`res_logs_30_runs.log` shows the consolidated result logs that were presented in the paper over 30 independently seeded runs.

### Inputs

`input/shaver-shell-full-all-classes-v2`: should contain the original dataset with three classes good, interrupted and double print

`input/network_good_80_iters`: should contain the pretrained stylegan3 model in .pkl format for the good class (similarly for `input/network_double_print_final` and `input/network_interrupted_final`)

`input/custom-defects-4cats`: should contain the open-set (novel defects) here divided in three classes (lines, missing letter, discoloration, horz. and vert. flip)

`input/sefa-output`: should contain the output of the data generation script that will be the input of the filtering script

`input/sefa-shaver-aug`: should contain the full augmented dataset after filtering with four classes (good, interrupted, double print, synth)

`input/sefa-utils`: contain data processing utility functions in utils.py

### Main Dependencies

python >= 3.7,
keras >= 2.6.0,
torch >= 1.11.0

### StyleGAN Training

The following command were used to train the StyleGAN model for the three classes. Please note that the first run for the "good" class produces a base pre-trained network that is further fine-tuned for all three classes.

```
#GOOD
!python stylegan3/train.py --outdir="./training-runs" --cfg=stylegan3-t --data="./shavershellgoodfull.zip" \
    --gpus=1 --batch=4 --gamma=8 --kimg=80 --freezed=10 --snap=5 --cbase=16384 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl

# GOOD 2nd run
!python stylegan3/train.py --outdir="./training-runs" --cfg=stylegan3-t --data="./shavershellgoodfull.zip" \
    --gpus=1 --batch=4 --gamma=8 --kimg=40 --freezed=10 --snap=5 --cbase=16384 \
    --resume="../input/network-good-80iters/network-good-80iters.pkl"

#DOUBLE PRINT
!python stylegan3/train.py --outdir="./training-runs" --cfg=stylegan3-t --data="./shavershelldoubleprintfull.zip" \
    --gpus=1 --batch=4 --gamma=8 --kimg=60 --freezed=10 --snap=5 --cbase=16384 \
    --resume="../input/network-good-80iters/network-good-80iters.pkl"
   
#INTERRUPTED
!python stylegan3/train.py --outdir="./training-runs" --cfg=stylegan3-t --data="./shavershellinterruptedfull.zip" \
    --gpus=1 --batch=4 --gamma=8 --kimg=100 --freezed=10 --snap=5 --cbase=16384 \
    --resume="../input/network-good-80iters/network-good-80iters.pkl"
```
### Links:

Code from the following repositories was used:

https://github.com/NVlabs/stylegan3,
https://github.com/genforce/sefa
