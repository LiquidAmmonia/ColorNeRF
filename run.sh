
exp_name=pear
LLFF_DIR='/home/chengyean/nerf_ws/nerf_data/nerf_llff_data/'${exp_name}

CUDA_IDS=3,4


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

CKPT_PATH=ckpts/$exp_name/last.ckpt
echo $CKPT_PATH
CUDA_VISIBLE_DEVICES=$CUDA_IDS python eval_color.py \
   --root_dir $LLFF_DIR \
   --dataset_name hfai_llff_ref \
   --scene_name $exp_name \
   --img_wh 640 360 \
   --N_importance 64 \
   --ckpt_path $CKPT_PATH \
   --local_run \
   --chunk 1024 \

stage2_exp_name=${exp_name}_stage2_color

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

echo "validating ..."

CKPT_PATH=ckpts/$stage2_exp_name/last.ckpt
echo $CKPT_PATH
python eval_color.py \
   --root_dir $LLFF_DIR \
   --dataset_name llff \
   --scene_name $stage2_exp_name \
   --img_wh 640 360 \
   --N_importance 64 \
   --ckpt_path $CKPT_PATH \
   --train_stage 2 \
   --chunk 4096 \
   --local_run \
   --focus_depth 1000


echo "******************************"
echo 'testing...'
CKPT_PATH=ckpts/$stage2_exp_name/last.ckpt
echo $CKPT_PATH
CUDA_VISIBLE_DEVICES=$CUDA_IDS python test.py \
   --root_dir $LLFF_DIR \
   --dataset_name llff \
   --scene_name $stage2_exp_name \
   --img_wh 640 360 \
   --N_importance 64 \
   --ckpt_path $CKPT_PATH \
   --train_stage 2 \
   --chunk 4096 \
   --local_run \
   # --color_class_T 0.38 \

