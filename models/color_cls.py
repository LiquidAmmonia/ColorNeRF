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
import sys
from pdb import set_trace as st

sys.path.append('/userhome/chengyean/ct2_sos')
from models.segm.model.utils import SoftEncodeAB, CIELAB, AnnealedMeanDecodeQ

from timm.utils import NativeScaler
from contextlib import suppress
import os

import collections
import skimage


from kornia.color import rgb_to_lab, lab_to_rgb

from tqdm import tqdm
import cv2


import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function



class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5):
        prior = torch.from_numpy(cielab.gamut.prior)

        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def __call__(self, ab_actual):
        self.weights = self.weights.to(ab_actual.device)
        return self.weights[ab_actual.argmax(dim=1, keepdim=True)]


class RebalanceLoss(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, data_input, weights):
        ctx.save_for_backward(weights)

        return data_input.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors

        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        grad_input = grad_output * weights

        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return grad_input, None



class ColorClassify():
    
    def __init__(self, 
                 class_rebal_lambda=0.5,
                 T=0,
                 ):
        # reigister the cielab class
        self.default_cielab = CIELAB()
        self.encode_ab = SoftEncodeAB(self.default_cielab)
    
        self.class_rebal_lambda = class_rebal_lambda
        self.get_class_weights = GetClassWeights(self.default_cielab,
                                                 lambda_=self.class_rebal_lambda)
        self.rebalance_loss = RebalanceLoss.apply
        # self.decode_q = AnnealedMeanDecodeQ(self.default_cielab, T=T)
        self.decode_q = AnnealedMeanDecodeQ(self.default_cielab, T=T)
        self.sigmoid = nn.Sigmoid()

    def _tensor_rgb_to_lab(self, rgb):
        lab = rgb_to_lab(rgb)
        
        img_l = lab[:, :1]
        img_ab = lab[:, 1:]

        return img_l, img_ab
    
    def get_q(self, rgb, rebalance=True):
        """
        rgb: [B, 3, H, W] in [0-1], used in gt conversion
        """
        img_l, img_ab = self._tensor_rgb_to_lab(rgb)
        img_q = self.encode_ab(img_ab)
        if rebalance:
            img_q = self.rebalance_loss(img_q, self.get_class_weights(img_q))
        return img_q
    
    def get_weighted_q(self, q_pred, q_gt):
        # q: [B, 313, H, W] from network output
        color_weights = self.get_class_weights(q_gt)
        q_pred = self.rebalance_loss(q_pred, color_weights)
        return q_pred
    
    def get_ab_infer(self, q, normalize=True):
        if normalize:
            q = self.sigmoid(q)
        ab = self.decode_q(q)
        return ab
        
    
if __name__ == "__main__":
    device = 'cuda:0'
    # img_path = '/home/chengyean/nerf_ws/nerf_data/nerf_llff_data/flower_gray/IMG_2962.JPG'
    img_path = '/userhome/chengyean/ct2_sos/models/out.png'

    def _tensor_rgb_to_lab(rgb):
        lab = rgb_to_lab(rgb)
        
        img_l = lab[:, :1]
        img_ab = lab[:, 1:]

        return img_l, img_ab

    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128), Image.Resampling.BILINEAR)
    img = np.array(img)
    
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
    img_torch = img_torch.unsqueeze(0).to(device)

    lab = rgb_to_lab(img_torch)
    img_l = lab[:, :1]
    img_ab = lab[:, 1:]
    # img_rgb = lab_to_rgb(torch.cat([img_l, img_ab], dim=1), clip=True)
    
    # out = img_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # out = (out * 255).astype(np.uint8)
    # Image.fromarray(out).save('lab_circle.png')
    
    model = ColorClassify(T=0)
    token_q = model.get_q(img_torch, rebalance=True)
    
    decode_ab = model.decode_q(token_q)
    
    out_lab = torch.cat([img_l, decode_ab], dim=1)
    out_rgb = lab_to_rgb(out_lab, clip=True)

    in_lab = torch.cat([img_l, img_ab], dim=1)
    in_rgb = lab_to_rgb(in_lab, clip=True)

    cat_out = torch.cat([in_rgb, out_rgb], dim=3)
    # visualize the tokens 
    q_img = cat_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    q_img = q_img * 255
    q_img = q_img.astype(np.uint8)

    out_dir = '/userhome/chengyean/ct2_sos/verbose'
    Image.fromarray(q_img).save(os.path.join(out_dir, 'q_img.png'))
    
    