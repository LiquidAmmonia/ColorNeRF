import os
from PIL import Image
import numpy as np
from skimage import color
import torch
import pickle
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.distributed
import warnings
import io

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress
import os

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate
import collections
import skimage
# from segm.engine import save_imgs

from pdb import set_trace as st
from tqdm import tqdm
import cv2

def rgb_to_lab(img):
    assert img.dtype == np.uint8
    return color.rgb2lab(img).astype(np.float32)

def numpy_to_torch(img):
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0))      # [c, h, w]
    return tensor.type(torch.float32)


def load_mask(mask_l_num):
    fp = open('/home/chengyean/nerf_ws/CT2/mask_prior.pickle', 'rb')
    # fp = open('/userhome/SUN_text2img/ImageNet/mask_prior.pickle', 'rb')
    L_dict = pickle.load(fp)
    mask_L = np.zeros((mask_l_num, 313)).astype(bool)
    for key in range(101):
        for ii in range(mask_l_num):
            start_key = ii * (100//mask_l_num)
            end_key = (ii + 1) * (100 // mask_l_num)
            if start_key <= key < end_key:
                mask_L[ii, :] += L_dict[key].astype(bool)
                break
    mask_L = mask_L.astype(np.float32)
    return mask_L


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", default='coco', type=str)
@click.option('--dataset_dir', default='/userhome/sjm/ImageNet',type=str)
@click.option("--im-size", default=256, type=int, help="dataset resize size")
@click.option("--crop-size", default=256, type=int)
@click.option("--window-size", default=256, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_large_patch16_384", type=str)  
# @click.option("--backbone", default="vit_tiny_patch16_384", type=str)       # try this, and freeze first several blocks.
@click.option("--decoder", default="mask_transformer", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option('--local_rank', type=int)
@click.option('--only_test', type=bool, default=True)
@click.option('--add_mask', type=bool, default=True)        # valid
@click.option('--partial_finetune', type=bool, default=False)       # compare validation, last finetune all blocks.
@click.option('--add_l1_loss', type=bool, default=True)            # add after classification.
@click.option('--l1_weight', type=float, default=10)
@click.option('--color_position', type=bool, default=True)     # add color position in color token.
@click.option('--change_mask', type=bool, default=False)        # change mask, omit the attention between color tokens.
@click.option('--color_as_condition', type=bool, default=False)     # use self-attn to embedding color tokens, and use color to represent patch tokens.
@click.option('--multi_scaled', type=bool, default=False)       # multi-scaled decoder.
@click.option('--downchannel', type=bool, default=False)        # multi-scaled, upsample+downchannel. (should be correct??)
@click.option('--add_conv', type=bool, default=True)       # add conv after transformer blocks.
@click.option('--before_classify', type=bool, default=False)        # classification at 16x16 resolution, and use CNN upsampler for 256x256, then use l1-loss.
@click.option('--l1_conv', type=bool, default=True)                # patch--upsample--> [B, 256x256, 180]--conv3x3-> [B, 256x256, 2]
@click.option('--l1_linear', type=bool, default=False)          # pacth: [B, 16x16, 180]---linear-> [B, 16x16, 2x16x16]
@click.option('--add_fm', type=bool, default=False)             # add Feature matching loss.
@click.option('--fm_weight', type=float, default=1)
@click.option('--add_edge', type=bool, default=False)       # add sobel-conv to extract edge.
@click.option('--edge_loss_weight', type=float, default=0.05)     # edge_loss_weight.
@click.option('--mask_l_num', type=int, default=4)          # mask for L ranges: 4, 10, 20, 50, 100
@click.option('--n_blocks', type=int, default=1)        # per block have 2 layers. block_num = 2
@click.option('--n_layers', type=int, default=2)
@click.option('--without_colorattn', type=bool, default=False)
@click.option('--without_colorquery', type=bool, default=False)
@click.option('--without_classification', type=bool, default=False)
@click.option('--mask_random', type=bool, default=False)
@click.option('--color_token_num', type=int, default=313)
@click.option('--sin_color_pos', type=bool, default=False)


@click.option('--data_dir', default='/userhome/chengyean/data/testcase_in_the_wild',type=str)
@click.option('--out_dir', default='/userhome/chengyean/data/ct2_refs',type=str)



def test_func(
    log_dir,
    data_dir,
    out_dir,
    with_fea,
    reshape_w, 
    reshape_h,
):

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # checkpoint_path = log_dir / 'checkpoint_epoch_0_psnrcls_22.8164_psnrreg_24.5049.pth'
    checkpoint_path = log_dir / 'checkpoint_epoch_10_mask_True.pth'
    # breakpoint()
    mask_l_num = 4
    
    net_kwargs = {'image_size': (256, 256), 'patch_size': 16, 'd_model': 1024, 'n_heads': 16, 'n_layers': 24, 'normalization': 'vit', 'backbone': 'vit_large_patch16_384', 'dropout': 0.0, 'drop_path_rate': 0.1, 'decoder': {'drop_path_rate': 0.0, 'dropout': 0.1, 'n_layers': 2, 'name': 'mask_transformer', 'add_l1_loss': True, 'color_position': True, 'change_mask': False, 'color_as_condition': False, 'multi_scaled': False, 'crop_size': 256, 'downchannel': False, 'add_conv': True, 'before_classify': False, 'l1_conv': True, 'l1_linear': False, 'add_edge': False, 'n_blocks': 1, 'without_colorattn': False, 'without_colorquery': False, 'without_classification': False, 'sin_color_pos': False}, 'n_cls': 313, 'partial_finetune': False}
    
    device = torch.device('cuda:0')
    model = create_segmenter(net_kwargs)
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    # load imgs.
    # img_path = '/userhome/chengyean/ARF-svox2/data/llff/flower_gray_origpose/images_4/image003.png'
    image_dir = data_dir
    save_dir = out_dir
    os.makedirs(save_dir, exist_ok=True)
    st()
    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        # resize to 256 x 256
        img = img.resize((256, 256), Image.Resampling.BILINEAR)
        img = np.array(img)

        l_resized = rgb_to_lab(img)[:, :, :1]
        ab_resized = rgb_to_lab(img)[:, :, 1:]  # np.float32
        original_l = l_resized[:, :, 0]

        l = original_l.reshape((256 * 256))

        mask_L = load_mask(mask_l_num)

        mask_p_c = np.zeros((256 ** 2, 313), dtype=np.float32)  # [256x256, 313]
        for l_range in range(mask_l_num):
            start_l1, end_l1 = l_range * (100//mask_l_num), (l_range+1) *(100//mask_l_num)
            if end_l1 == 100:
                index_l1 = np.where((l >= start_l1) & (l <= end_l1))[0]
            else:
                index_l1 = np.where((l >= start_l1) & (l < end_l1))[0]
            mask_p_c[index_l1, :] = mask_L[l_range, :]

        mask = torch.from_numpy(mask_p_c)  # [256*256, 313]

        img_l = numpy_to_torch(l_resized)
        img_ab = numpy_to_torch(ab_resized)
        mask, img_l, img_ab = mask.unsqueeze(0), img_l.unsqueeze(0), img_ab.unsqueeze(0)

        model_without_ddp = model
        if hasattr(model, "module"):
            model_without_ddp = model.module

        with torch.no_grad():
            img_l = img_l.to(device)
            img_ab = img_ab.to(device)
            mask = mask.to(device)
            ab_pred, q_pred, q_actual, out_feature, patches = model_without_ddp.inference(img_l, img_ab, mask, appli=True)
            # ab_pred, q_pred, q_actual, out_feature = model_without_ddp.inference(img_l, img_ab, mask, appli=True)
            # breakpoint()
            # st()
            save_imgs(img_l, img_ab, ab_pred, img_name, save_dir, reshape_w, reshape_h)
            # save_imgs(img_l, img_ab, nn.Softmax(dim=1)(out_feature) * 110, 'fea_'+img_name, save_dir)
            # save feature maps
            if with_fea:
                np.save(os.path.join(save_dir, img_name.replace('.png', '.npy')), patches.detach().cpu().numpy())
                np.save(os.path.join(save_dir, img_name.replace('.png', '_out_feature.npy')), out_feature.detach().cpu().numpy())
            # st()
            # m = 1



def lab_to_rgb(img):
    assert img.dtype == np.float32
    return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)

def save_imgs(img_l, img_ab, ab_pred, filenames, dir, reshape_w, reshape_h):
    img_lab = torch.cat((img_l, ab_pred.detach()), dim=1).cpu()
    batch_size = img_lab.size(0)
    fake_rgb_list, real_rgb_list, only_rgb_list = [], [], []
    for j in range(batch_size):
        img_lab_np = img_lab[j].numpy().transpose(1, 2, 0)      # np.float32
        img_rgb = lab_to_rgb(img_lab_np)        # np.uint8      # [0-255]
        fake_rgb_list.append(img_rgb)

        img_path = os.path.join(dir, filenames)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if not (reshape_h == 0 and reshape_w == 0):
                img_rgb = cv2.resize(img_rgb, (reshape_w, reshape_h), interpolation=cv2.INTER_CUBIC)
            skimage.io.imsave(img_path, img_rgb.astype(np.uint8))
            # print('successful save imgs. ')

if __name__ == '__main__':
    test_func()







