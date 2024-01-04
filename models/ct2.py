

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

from models.segm.model.factory import create_segmenter

from timm.utils import NativeScaler
from contextlib import suppress
import os

import collections
import skimage

from kornia.color import rgb_to_lab, lab_to_rgb

from pdb import set_trace as st
from tqdm import tqdm
import cv2

import torch.nn as nn

import torch.nn.functional as F

class CT2Wrapper(nn.Module):
    def __init__(self, device='cpu', root_path=None, 
                 model_name=None):
        super().__init__()
        # self.device = device
        self.root_path = 'models/segm' if root_path is None else root_path
        print('root_path', self.root_path)
        if model_name is None:
            model_name = 'checkpoint_epoch_10_mask_True.pth'
        self.model_name = model_name
            
        self._get_base_kwargs()
        self._load_model()
        self.model = self.model
        self.mask_L = self._load_mask_l()
        
    def _get_base_kwargs(self):
        self.net_kwargs = {
            'image_size': (256, 256), 
            'patch_size': 16, 
            'd_model': 1024, 
            'n_heads': 16, 
            'n_layers': 24, 
            'normalization': 'vit', 
            'backbone': 'vit_large_patch16_384', 
            'dropout': 0.0, 
            'drop_path_rate': 0.1, 
            'decoder': {
                'drop_path_rate': 0.0, 
                'dropout': 0.1, 
                'n_layers': 2, 
                'name': 'mask_transformer', 
                'add_l1_loss': True, 
                'color_position': True, 
                'change_mask': False, 
                'color_as_condition': False, 
                'multi_scaled': False, 
                'crop_size': 256, 
                'downchannel': False, 
                'add_conv': True, 
                'before_classify': False, 
                'l1_conv': True, 
                'l1_linear': False, 
                'add_edge': False, 
                'n_blocks': 1, 
                'without_colorattn': False, 
                'without_colorquery': False,
                'without_classification': False, 
                'sin_color_pos': False
                }, 
            'n_cls': 313, 
            'partial_finetune': False,
            'root_dir': self.root_path
            }
        
    def _load_model(self):
        
        self.model_path = os.path.join(self.root_path, 'vit-large', 
                                       self.model_name)
        
        print("Loading model from {}".format(self.model_path))
        
        self.model = create_segmenter(self.net_kwargs)
        # checkpoint = torch.load(self.model_path, map_location=self.device)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model"])

        if hasattr(self.model, "module"):
            self.model = self.model.module
            
        for param in self.model.parameters():
            param.requires_grad = False
            
        
    def _load_mask_l(self, mask_l_num=4):
        file_path = os.path.join(self.root_path, 'mask_prior.pickle')
        fp = open(file_path, 'rb')
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
        return torch.from_numpy(mask_L)
           
    def _generate_mask(self, l, mask_l_num=4):
        l = l.reshape((256 * 256))
        mask_p_c = torch.zeros((256 ** 2, 313), dtype=torch.float)  # [256x256, 313]
        for l_range in range(mask_l_num):
            start_l1, end_l1 = l_range * (100//mask_l_num), (l_range+1) *(100//mask_l_num)
            if end_l1 == 100:
                index_l1 = torch.where((l >= start_l1) & (l <= end_l1))[0]
            else:
                index_l1 = torch.where((l >= start_l1) & (l < end_l1))[0]
            mask_p_c[index_l1, :] = self.mask_L[l_range, :]

        mask = mask_p_c  # [256*256, 313]
        return mask

    def _tensor_rgb_to_lab(self, rgb):
        lab = rgb_to_lab(rgb)
        
        img_l = lab[:, :1]
        img_ab = lab[:, 1:]

        return img_l, img_ab
    
    def _tensor_lab_to_rgb(self, img_l, img_ab):
        img_lab = torch.cat([img_l, img_ab], dim=1)
        img_rgb = lab_to_rgb(img_lab, clip=True)
        
        return img_rgb
    
    def inference(self, input):
        
        with torch.no_grad():
            input = F.interpolate(input, size=(256, 256), mode='bilinear', align_corners=False)
            # input shape: (1, 3, 256, 256)
            img_l, img_ab = self._tensor_rgb_to_lab(input)
            # input scale is 0-1 but img_l is 0-100, img_ab is -128-127
            
            mask = self._generate_mask(img_l).to(input.device)
            mask = mask.unsqueeze(0)
            
            ab_pred, _, _, _, _ = \
                self.model.inference(img_l, img_ab, mask, appli=True)
            out = self._tensor_lab_to_rgb(img_l, ab_pred)
        
        return out
    
if __name__ == "__main__":
    device = 'cuda:0'
    img_path = '/home/chengyean/nerf_ws/nerf_data/nerf_llff_data/flower_gray/IMG_2962.JPG'
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    img = np.array(img)
    
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
    img_torch = img_torch.unsqueeze(0).to(device)
    
    # lab = rgb_to_lab(img_torch)
    # img_l = lab[:, :1]
    # img_ab = lab[:, 1:]
    # img_rgb = lab_to_rgb(torch.cat([img_l, img_ab], dim=1), clip=True)
    
    # out = img_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # out = (out * 255).astype(np.uint8)
    # Image.fromarray(out).save('lab_circle.png')
    
    model = CT2Wrapper(device)
    out = model.inference(img_torch)
    
    # save the output
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).astype(np.uint8)
    Image.fromarray(out).save('out.png')
    
