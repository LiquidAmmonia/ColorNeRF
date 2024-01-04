
import os, sys

sys.path.append('/userhome/chengyean/ct2_sos')
from models.zhang_color.options.train_options import TrainOptions
from models.zhang_color.models import create_model
# from util.visualizer import save_images
# from util import html

# import string
import torch
# import torchvision
# import torchvision.transforms as transforms

from models.zhang_color.util import util
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as st
import cv2

from skimage import transform

class ZhangColorWrapper(nn.Module):
    def __init__(self, 
                 device='cpu', 
                 root_path=None, 
                 model_name=None, 
                 mode='random'):
        super().__init__()
        self.root_path = 'models/zhang_color/checkpoints' if root_path is None else root_path
        self.model_name = 'latest_net_G.pth' if model_name is None else model_name # not used for now
        self.init_device = device
        self._get_args()

        self._load_model()
        self.sample_p = .125
        self.num_pts = 5

        self.to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', 'mask']
        
        self.mode = mode 
        self.hints = None
        if self.mode == 'fix':
            print("Using fixed hint")
            # self.register_hint()
            
    def register_imgset(self, ffr_wrapper, img_paths, img_wh, verbose=False):
        self.ffr_wrapper = ffr_wrapper
        self.img_paths = img_paths
        self.img_wh = img_wh
        verbose_imgs =  self.register_hint(verbose)
        
        return verbose_imgs
    
    def _get_args(self):
        # self.opt = TrainOptions().parse()
        # self.opt.load_model = True
        # self.opt.num_threads = 1   # test code only supports num_threads = 1
        # self.opt.batch_size = 1  # test code only supports batch_size = 1
        # self.opt.display_id = -1  # no visdom display
        # self.opt.phase = 'val'
        # self.opt.serial_batches = True
        # self.opt.aspect_ratio = 1.
        opt_dict = {'batch_size': 1, 'loadSize': 256, 'fineSize': 176, 'input_nc': 1, 'output_nc': 2, 'ngf': 64, 'ndf': 64, 'which_model_netD': 'basic', 'which_model_netG': 'siggraph', 'n_layers_D': 3, 'gpu_ids': [], 'name': 'siggraph_retrained', 'dataset_mode': 'aligned', 'model': 'pix2pix', 'which_direction': 'AtoB', 'num_threads': 1, 'checkpoints_dir': './checkpoints', 'norm': 'batch', 'serial_batches': True, 'display_winsize': 256, 'display_id': -1, 'display_server': 'http://localhost', 'display_port': 8097, 'no_dropout': False, 'max_dataset_size': float("inf"), 'resize_or_crop': 'resize_and_crop', 'no_flip': False, 'init_type': 'normal', 'verbose': False, 'suffix': '', 'ab_norm': 110.0, 'ab_max': 110.0, 'ab_quant': 10.0, 'l_norm': 100.0, 'l_cent': 50.0, 'mask_cent': 0.5, 'sample_p': 1.0, 'sample_Ps': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'results_dir': './results/', 'classification': False, 'phase': 'val', 'which_epoch': 'latest', 'how_many': 200, 'aspect_ratio': 1.0, 'load_model': True, 'half': False, 'display_freq': 10000, 'display_ncols': 5, 'update_html_freq': 10000, 'print_freq': 200, 'save_latest_freq': 5000, 'save_epoch_freq': 1, 'epoch_count': 0, 'niter': 100, 'niter_decay': 100, 'beta1': 0.9, 'lr': 0.0001, 'no_lsgan': False, 'lambda_GAN': 0.0, 'lambda_A': 1.0, 'lambda_B': 1.0, 'lambda_identity': 0.5, 'pool_size': 50, 'no_html': False, 'lr_policy': 'lambda', 'lr_decay_iters': 50, 'avg_loss_alpha': 0.986, 'isTrain': True, 'A': 23.0, 'B': 23.0, 'dataroot': '/userhome/chengyean/colorization-pytorch/llff_data/flower'}

        import argparse
        self.opt = argparse.Namespace(**opt_dict)

        # modifications
        self.opt.checkpoints_dir = self.root_path
        # convert device to gpu_ids
        # self.opt.gpu_ids = [int(self.device.split(':')[-1])]

    def _load_model(self):
        self.model = create_model(self.opt, self.init_device)
        self.model.setup(self.opt)
        self.model.eval()
    
    def _read_img_to_tensor(self, img_path):
        img_0 = cv2.imread(img_path) # RGB image
        img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
        img_0_tensor = torch.from_numpy(img_0)
        img_0_tensor = img_0_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.
        return img_0_tensor, img_0
    
    def _read_img_to_tensor_from_ffr(self, img_path):
        # img_0 = cv2.imread(img_path) # RGB image
        img = self.ffr_wrapper.decode_img(img_path)
        # img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB) # included in decode_img
        img = cv2.resize(np.array(img), self.img_wh, interpolation=cv2.INTER_LANCZOS4)
        img_0_tensor = torch.from_numpy(img)
        img_0_tensor = img_0_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.
        return img_0_tensor, img
        
    def register_hint(self, verbose=False):
        # root_dir = '/userhome/chengyean/nerf_llff_data_images/flower/images_8'
        # img_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        # img_list = sorted(img_list)
        img_list = self.img_paths
        img_0_path = img_list[0]
        # img_tensor, img_0 = self._read_img_to_tensor(img_0_path)
        img_tensor, img_0 = self._read_img_to_tensor_from_ffr(img_0_path)
        
        hint_0 = util.get_colorization_data([img_tensor],
                                              self.opt,
                                              ab_thresh=0.,
                                              p=self.sample_p, 
                                              num_points=self.num_pts)
        # visualize the hint
        hint_B, mask_B = hint_0['hint_B'], hint_0['mask_B']
        save_dir = '/userhome/chengyean/ct2_sos/verbose/hint_vis'
        # os.makedirs(save_dir, exist_ok=True)
        def save_img(img, save_name):
            img = img.squeeze(0).permute(1, 2, 0).numpy() * 255.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, save_name), img)
        
        # save_img(hint_B, 'hint_B.png')
        if verbose:
            save_img(mask_B * img_tensor, 'mask_B.png')
        
        self.hints = [hint_B]
        sift = cv2.SIFT_create()
        # st()
        
        fake_reg_0, hint_0 = self.validate_color_results(img_tensor, hint_B)
        if verbose:
            save_img(fake_reg_0.cpu(), 'fake_reg_src.png')
            save_img(hint_0.cpu(), 'hint_src.png')
        
        for img_path in img_list[1:]:
            dst_img_tensor, dst_img = self._read_img_to_tensor_from_ffr(img_path)
            homo_mat = self._get_transmat(sift, img_0, dst_img)
            
            warped_hint_np = transform.warp(hint_B.squeeze(0).permute(1, 2, 0).numpy(), homo_mat.inverse, mode='constant')
            warped_hint = torch.from_numpy(warped_hint_np).permute(2, 0, 1).unsqueeze(0).float()
            
            self.hints.append(warped_hint)
            if verbose:
                warped_hint_mask_np = transform.warp(mask_B.squeeze(0).permute(1, 2, 0).numpy(), homo_mat.inverse, mode='constant')
                warped_hint_mask = torch.from_numpy(warped_hint_mask_np).permute(2, 0, 1).unsqueeze(0).float()

                img_name = os.path.basename(img_path)
                save_img(warped_hint_mask, 'warped_hint_mask' + img_name)
                fake_reg, hint = self.validate_color_results(dst_img_tensor, warped_hint)
                save_img(fake_reg.cpu(), 'fake_reg' + img_name)
                save_img(hint.cpu(), 'hint_' + img_name)
        
        return fake_reg_0.cpu(), hint_0.cpu()
            
    def validate_color_results(self, input, hint):
        with torch.no_grad():
            input = F.interpolate(input, 
                                  size=(256, 256), 
                                  mode='bilinear', 
                                  align_corners=False)
            hint = F.interpolate(hint,
                                 size=(256, 256), 
                                 mode='bilinear', 
                                 align_corners=False)

            data = util.get_colorization_data_withmask([input], self.opt, hint)
            # st()

            self.model.set_input(data)
            self.model.test(False)  # True means that losses will be computed
            visuals = util.get_subset_dict(self.model.get_current_visuals(), 
                                           self.to_visualize)

        return visuals['fake_reg'], visuals['hint']
        
    
    def _get_transmat(self, sift, src_img, dst_img):
        kp0, des0 = sift.detectAndCompute(src_img, None)
        kp_i, des_i = sift.detectAndCompute(dst_img, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des0,des_i,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        src_pts = np.float32([ kp0[m.queryIdx].pt for m in good ]).reshape(-1,2)
        dst_pts = np.float32([ kp_i[m.trainIdx].pt for m in good ]).reshape(-1,2)

        homo_mat = transform.estimate_transform('projective', src_pts, dst_pts)
        return homo_mat
    
        
        
    def inference(self, input, verbose=False, debug=False):
        if self.mode == 'random':
            return self.random_inference(input, verbose, debug)
        elif self.mode == 'fix':
            return self.fix_inference(input, verbose, debug)

    def fix_inference(self, input, verbose=False, debug=False):
        """
        Fix inference, 
        """
        assert self.hints is not None, "Please register hint first"
        assert type(input) == list, "Please input a rgb_gt, select_inds, and idx as a list"
        input, select_inds, idx = input
        
        # def _grid_sample(img, inds):
        #     return F.grid_sample(img.permute(2, 0, 1).unsqueeze(0),
        #                          inds.unsqueeze(0), 
        #                          align_corners=True).squeeze(0).permute(1, 2, 0)
        
        # def _flatten(img):
        #     return img.view(-1, img.shape[-1])
        # st()
        hint = self.hints[idx].to(input.device)
        hint_patch = F.grid_sample(hint, select_inds ,align_corners=True)
        
        with torch.no_grad():
            input = F.interpolate(input, 
                                  size=(256, 256), 
                                  mode='bilinear', 
                                  align_corners=False)
            hint = F.interpolate(hint_patch,
                                 size=(256, 256), 
                                 mode='bilinear', 
                                 align_corners=False)

            data = util.get_colorization_data_withmask([input], self.opt, hint)
            # st()

            self.model.set_input(data)
            self.model.test(False)  # True means that losses will be computed
            visuals = util.get_subset_dict(self.model.get_current_visuals(), 
                                           self.to_visualize)
            
        if verbose:
            if not debug:
                return visuals['fake_reg'], visuals['hint']
            else:
                return visuals['fake_reg'], visuals['hint'], visuals['mask']
        else: 
            return visuals['fake_reg'], None

        
    def random_inference(self, input, verbose=False, debug=False):
        """
        randomly sample hints from the gt image. 
        This sampling will change along with the patch selection
        """
        with torch.no_grad():
            input = F.interpolate(input, 
                                  size=(256, 256), 
                                  mode='bilinear', 
                                  align_corners=False)
            input = [input] # spectial case for Zhang's code

            data = util.get_colorization_data(input,
                                              self.opt,
                                              ab_thresh=0.,
                                              p=self.sample_p, 
                                              num_points=self.num_pts)
            # st()

            self.model.set_input(data)
            self.model.test(False)  # True means that losses will be computed
            visuals = util.get_subset_dict(self.model.get_current_visuals(), 
                                           self.to_visualize)

        if verbose:
            if not debug:
                return visuals['fake_reg'], visuals['hint']
            else:
                return visuals['fake_reg'], visuals['hint'], visuals['mask']
        else: 
            return visuals['fake_reg'], None


    

if __name__ == '__main__':
    device = 'cuda:0'
    root_path = '/userhome/chengyean/colorization-pytorch/llff_data/flower/images_8'
    model = ZhangColorWrapper(device=device, mode='random')
    out_dir = '/userhome/chengyean/ct2_sos/verbose/zhang'
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    for img_name in os.listdir(root_path):
        # if 'gray' in img_name:
        img_path = os.path.join(root_path, img_name)
        from PIL import Image as Image
        print(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img = img / 255.0

        # model.to('cuda:0')
        # img = img.to(device)
        out, hint, mask = model.inference(img, verbose=True, debug=True)

        out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        out.save(os.path.join(out_dir, img_name))

        out = hint.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        out.save(os.path.join(out_dir, 'hint_' + img_name))

        out = mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        out.save(os.path.join(out_dir, 'mask_' + img_name))

        # ds_img = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=False)
        # ds_img = F.interpolate(ds_img, size=(256, 256), mode='bilinear', align_corners=False)
        # out, hint = model.inference(ds_img, verbose=True)

        # out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # out = np.clip(out, 0, 1)
        # out = (out * 255).astype(np.uint8)
        # out = Image.fromarray(out)
        # out.save(os.path.join(out_dir, 'ds_'+img_name))

        # out = hint.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # out = np.clip(out, 0, 1)
        # out = (out * 255).astype(np.uint8)
        # out = Image.fromarray(out)
        # out.save(os.path.join(out_dir, 'ds_hint_' + img_name))
    
    
    # model = ZhangColorWrapper(device=device, mode='fix')