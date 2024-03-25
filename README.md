# Colorizing Monochromatic Radiance Fields

This is the official codebase of [Colorizing Monochromatic Radiance Fields](https://liquidammonia.github.io/color-nerf/) (AAAI 2024, Oral presentation).


## Quick start

### Env setup
Create conda environment from env.yaml
```
conda create -f env.yaml
```
Note: some packages are not used in this demo.

### Data preparation
We use LLFF dataset for training and testing. Please download the dataset from [the official website](https://github.com/Fyusion/LLFF)

### Train
There are two training stages: (1) train a monochromatice NeRF model, and (2) train a colorized model.
```
# stage 1
CUDA_VISIBLE_DEVICES=$CUDA_IDS python train_color.py \
   --dataset_name hfai_llff_ref \
   --root_dir $LLFF_DIR \
   --N_importance 64 --img_wh 640 360 \
   --num_epochs 40 --batch_size 8192 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name $exp_name \
   --loss_type color \
   --num_gpus 2 \
   --num_workers 32 \
   --not_use_patch \
   --normalize_illu \
   --local_run \
   --val_sanity_epoch 1
```
```
# stage 2
CUDA_VISIBLE_DEVICES=$CUDA_IDS python train_color.py \
   --dataset_name llff \
   --root_dir $LLFF_DIR \
   --weight_path ckpts/${exp_name}/last.ckpt \
   --N_importance 64 --img_wh 640 360 \
   --num_epochs 30 --batch_size 4096 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 30 40 --decay_gamma 0.1 \
   --exp_name $stage2_exp_name \
   --loss_type color \
   --num_gpus 2 \
   --num_workers 64 \
   --use_patch \
   --teacher_model 'ct2' \
   --train_stage 2 \
   --patch_sample_method central \
   --local_run \
   --use_color_hist \
   --use_color_class_loss \
   --not_color_hist_force_accept \
   --color_hist_thres 0.90 \
   --chunk 8192 \
   --use_tv_loss 
```

You can refer to a comprehensive training and testing script in `run.sh`.
```
bash run.sh
```

## Acknowledgement

The codebase of NeRF is derived from [NeRF-SOS](https://github.com/VITA-Group/NeRF-SOS). We thank the authors for their great work.

The codebase of `models/segm` is derived from [CT2](https://github.com/shuchenweng/CT2) by @Shuchen Weng.


The codebase of `models/lcoder` is derived from [L-CoDer](https://github.com/changzheng123/L-CoDer) by @Zheng Chang.

The codebase of `models/zhang_color` is derived from [colorization](https://github.com/richzhang/colorization) by @Richard Zhang.


## Citation

The paper bibtex is as follows
```
@inproceedings{cheng2024colornerf,
  author    = {Yean Cheng, Renjie Wan, Shuchen Weng, Chengxuan Zhu, Yakun Chang, Boxin Shi},
  title     = {Colorizing Monochromatic Radiance Fields},
  journal   = {AAAI},
  year      = {2024},
}
```
