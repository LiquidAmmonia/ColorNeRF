# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys

from pathlib import Path
from collections import OrderedDict

# from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

# from datasets import TextDataset,TextDataset_pair

# from models.lcoder.engine_for_colorization import train_one_epoch, evaluate
# from models.lcoder.utils import NativeScalerWithGradNormCount as NativeScaler

sys.path.append('/userhome/chengyean/ct2_sos')
from models.lcoder_model.utils import load_state_dict, get_colorization_data, lab2rgb
# from scipy import interpolate
import models.lcoder_model.modeling_colorization

import torch.nn.functional as F
import torch.nn as nn

from timm.models import create_model

from pdb import set_trace as st

class LCoderWrapper(nn.Module):
    def __init__(self, device='cpu',
                       cap='a red flower with green leaves', 
                       root_path=None, 
                       model_name=None):
        super().__init__()
        self.root_path = 'models/lcoder_model' if root_path is None else root_path
        print('root_path', self.root_path)
        if model_name is None:
            model_name = 'checkpoint-best.pth'
        self.model_name = model_name

        self.model_path = os.path.join(self.root_path, self.model_name)

        # print("The cap for l-coder model is: ", cap)
        self.cap = cap

        # self.args = self._get_args()

        self._load_model()
        self.model.eval()
        self.model.to(device)

    def _get_args(self):

        parser = argparse.ArgumentParser('MAE colorization and evaluation script for image classification', add_help=False)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--update_freq', default=1, type=int)
        parser.add_argument('--save_ckpt_freq', default=20, type=int)

        # Model parameters
        parser.add_argument('--model', default='colorization_mae_large_patch16_224_fusion_whole_up', type=str, metavar='MODEL',
                            help='Name of model to train')

        parser.add_argument('--input_size', default=224, type=int,
                            help='images input size')

        # parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
        #                     help='Dropout rate (default: 0.)')
        # parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
        #                     help='Attention dropout rate (default: 0.)')
        parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                            help='Drop path rate (default: 0.1)')

        parser.add_argument('--disable_eval', action='store_true', default=False)

        parser.add_argument('--model_ema', action='store_true', default=False)
        parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
        parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

        # Optimizer parameters
        parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            help='Optimizer (default: "adamw"')
        parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')
        parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: None, use opt default)')
        parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                            help='Clip gradient norm (default: None, no clipping)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help='weight decay (default: 0.05)')
        parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")

        parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                            help='learning rate (default: 1e-3)')
        parser.add_argument('--layer_decay', type=float, default=0.75)

        parser.add_argument('--warmup_lr', type=float, default=1e-9, metavar='LR',
                            help='warmup learning rate (default: 1e-6)')
        parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

        parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
        parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                            help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

        # Augmentation parameters
        parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                            help='Color jitter factor (default: 0.4)')
        parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
        parser.add_argument('--smoothing', type=float, default=0.1,
                            help='Label smoothing (default: 0.1)')
        parser.add_argument('--train_interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

        # Evaluation parameters
        parser.add_argument('--crop_pct', type=float, default=None)

        # * Random Erase params
        parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
        parser.add_argument('--remode', type=str, default='pixel',
                            help='Random erase mode (default: "pixel")')
        parser.add_argument('--recount', type=int, default=1,
                            help='Random erase count (default: 1)')
        parser.add_argument('--resplit', action='store_true', default=False,
                            help='Do not random erase first (clean) augmentation split')

        # * Mixup params
        parser.add_argument('--mixup', type=float, default=0,
                            help='mixup alpha, mixup enabled if > 0.')
        parser.add_argument('--cutmix', type=float, default=0,
                            help='cutmix alpha, cutmix enabled if > 0.')
        parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                            help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
        parser.add_argument('--mixup_prob', type=float, default=1.0,
                            help='Probability of performing mixup or cutmix when either/both is enabled')
        parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                            help='Probability of switching to cutmix when both mixup and cutmix enabled')
        parser.add_argument('--mixup_mode', type=str, default='batch',
                            help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

        # * Finetuning params
        parser.add_argument('--finetune', default='', help='finetune from checkpoint')
        parser.add_argument('--model_key', default='model|module', type=str)
        parser.add_argument('--model_prefix', default='', type=str)
        parser.add_argument('--init_scale', default=0.001, type=float)
        parser.add_argument('--use_mean_pooling', action='store_true')
        parser.set_defaults(use_mean_pooling=True)
        parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

        # Dataset parameters
        parser.add_argument('--data_path', default='/data2/cz2021/MSCOCO', type=str,
                            help='dataset path')
        # parser.add_argument('--eval_data_path', default=None, type=str,
        #                     help='dataset path for evaluation')
        parser.add_argument('--nb_classes', default=1000, type=int,
                            help='number of the classification types')
        parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

        parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                            type=str, help='ImageNet dataset path')
        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--log_dir', default=None,
                            help='path where to tensorboard log')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--resume', default='',
                            help='resume from checkpoint')
        parser.add_argument('--auto_resume', action='store_true')
        parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
        parser.set_defaults(auto_resume=True)

        parser.add_argument('--save_ckpt', action='store_true')
        parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
        parser.set_defaults(save_ckpt=True)

        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', action='store_true',
                            help='Perform evaluation only')
        parser.add_argument('--test', action='store_true',
                            help='Perform test only')
        parser.add_argument('--dist_eval', action='store_true', default=False,
                            help='Enabling distributed evaluation')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--pin_mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
        parser.set_defaults(pin_mem=True)

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')

        parser.add_argument('--enable_deepspeed', action='store_true', default=False)


        return parser.parse_args()


    def _load_model(self):
        """Load model from checkpoint, reigister in self.model"""
        model = create_model(
            'colorization_mae_large_patch16_224_fusion_whole_up',# self.args.model,
            pretrained=False,# 这个false是不从timm加载模型
            drop_path_rate=0.1, #self.args.drop_path
            drop_block_rate=None,
        )

        patch_size = model.encoder.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        # self.args.window_size = (224 // patch_size[0], 
        #                          224 // patch_size[1])
        # self.args.patch_size = patch_size

        checkpoint = torch.load(self.model_path, map_location='cpu')

        print("Load ckpt from %s" % self.model_path)
        checkpoint_model = None
        args_model_key = 'model|module'
        for model_key in args_model_key.split('|'): # default='model|module'
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint


        # 去掉head中shape不同的权重
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # 新建权重字典
        
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()

        for key in all_keys:
            if key.startswith('patch_embed.'):
                new_dict['encoder.'+ key] = checkpoint_model[key]
            elif key.startswith('pos_embed'):
                new_dict['encoder.'+key] = checkpoint_model[key]
            elif key.startswith('blocks.'):
                new_dict['encoder.'+key] = checkpoint_model[key]
            elif key.startswith('norm.'):
                new_dict['encoder.'+key] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
       
        
        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        load_state_dict(model, checkpoint_model, prefix='')
        # model.load_state_dict(checkpoint_model, strict=False)

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False


    def inference(self, input, cap=None):
        """
        img: tensor of size like the other wrapper
        """
        with torch.no_grad():
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            # input shape: (1, 3, 224, 224)
            color_data = get_colorization_data(input)
            img_l = color_data['A'] # 取值范围[-1,1]
            img_ab = color_data['B'] # 取值范围[-1,1]
            # input scale is 0-1 but img_l is 0-100, img_ab is -128-127
            # st()
            if cap is not None:
                ab_pred, _ = self.model(img_l.repeat(1, 3, 1, 1), [cap])
            else:
                ab_pred, _ = self.model(img_l.repeat(1, 3, 1, 1), [self.cap])
            out = lab2rgb(torch.cat((img_l, ab_pred), dim=1))
        
        return out

if __name__ == '__main__':
    # root_path = '/userhome/chengyean/ARF-svox2/data/llff/flower_gray_origpose/images_8'
    # root_path = '/userhome/chengyean/nerf_llff_data_images/apple8/images'
    root_path = '/userhome/chengyean/nerf_llff_data_images/balloon3/images'
    # root_path = '/userhome/chengyean/ct2_sos/verbose'
    cuda_id = 'cuda:0'
    out_dir = 'lcoder_results'
    model = LCoderWrapper(device=cuda_id, 
                          root_path='/userhome/chengyean/ct2_sos/models/lcoder_model', 
                          model_name='checkpoint-best.pth')
    model = model.to(cuda_id)
    cap_ls = [
        # 'red flower in green leave',
        # 'green apple on black table',
        # 'red apple on black table',
        # 'purple balloon on yellow wall',
        'green balloon on white wall',
        'blue balloon on white wall',
        # 'orange balloon on white wall',
        # 'brown balloon on white wall',
        # 'pink balloon on yellow wall',
        ]
    for cap in cap_ls:
        cap_dir = os.path.join(out_dir, cap.replace(' ', '_'))
        os.makedirs(cap_dir, exist_ok=True)
        for img_name in os.listdir(root_path):
            # if 'gray' in img_name:
            img_path = os.path.join(root_path, img_name)
            from PIL import Image as Image
            print(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            img = img / 255.0

            downsized_img = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=False)
            # # scale back to 224 x 224
            downsized_img = F.interpolate(downsized_img, size=(224, 224), mode='bilinear', align_corners=False)
            # cat 2x2 identical images together
            # downsized_img_ = torch.cat([downsized_img, downsized_img], dim=2)
            # downsized_img = torch.cat([downsized_img_, downsized_img_], dim=3)
            # st()
            
            # model.to('cuda:0')
            img = img.to(cuda_id)
            out = model.inference(img, cap=cap)
            out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            out = np.clip(out, 0, 1)
            out = (out * 255).astype(np.uint8)
            out = Image.fromarray(out)
            out.save(os.path.join(cap_dir, img_name))

            # model.to('cuda:0')
            img = downsized_img.to(cuda_id)
            out = model.inference(img, cap=cap)
            out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            out = np.clip(out, 0, 1)
            out = (out * 255).astype(np.uint8)
            out = Image.fromarray(out)
            out.save(os.path.join(cap_dir, 'ds_'+img_name))
            
            # st()
    # img_path = '/userhome/chengyean/lcoder/MAE_new/flower_gray/IMG_2962.JPG'
    # from PIL import Image as Image
    # img = Image.open(img_path).convert('RGB')
    # img = np.array(img)
    # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    # img = img / 255.0

    # model = LCoderWrapper(root_path='models/lcoder_model', model_name='checkpoint-best.pth')
    # model.to('cuda:0')
    # img = img.to('cuda:0')

    # out = model.inference(img)
    # out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # out = (out * 255).astype(np.uint8)
    # out = Image.fromarray(out)
    # out.save('out.jpg')
