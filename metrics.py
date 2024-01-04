import torch

from skimage.metrics import structural_similarity as m_ssim
import cv2
import numpy as np
import os

from pdb import set_trace as st

from models.segm.metrics import calculate_fid

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    psnr_ = -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))
    return psnr_.item()

def psnr_tensor(image_pred, image_gt, valid_mask=None, reduction='mean'):
    psnr_ = -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))
    return psnr_

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    # image_pred_np = image_pred.numpy()
    def permute(img):
        # convert [1, 3, h, w] to [h,w,3]
        out = img.permute(0, 2, 3, 1).squeeze(0).numpy()
        return out 
    # st()
    dssim_ = m_ssim(permute(image_pred), permute(image_gt), channel_axis=2) # dissimilarity in [0, 1]
    
    return dssim_

def lpips(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    from lpips import LPIPS
    lpips_module = LPIPS(net='alex').cuda()
    lpips_ = lpips_module(image_pred, image_gt)
    
    return lpips_.item()


def image_colorfulness(image):
    # image: np array of shape (h, w, 3)
    (B, G, R) = cv2.split(image.astype('float'))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R+G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def my_calculate_fid(pred_path, img_path):
    # pred_path: path to the folder containing the predicted images
    
    return calculate_fid(pred_path, img_path)
    
    
if __name__ == '__main__':
    gt_dir = '/userhome/chengyean/ct2_sos/test_results/hfai_llff_ref/purple1_stage2_color_ft_T038/gt'
    pred_dir = '/userhome/chengyean/ct2_sos/test_results/hfai_llff_ref/purple1_stage2_color_ft_T038/pred'
    breakpoint()
    out = my_calculate_fid(pred_dir, gt_dir)
    # os.system(f'python -m pytorch_fid {gt_dir} {pred_dir}')
    