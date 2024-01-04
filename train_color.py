
import warnings
warnings.simplefilter("ignore", UserWarning)
import os

from pytorch_lightning.accelerators import accelerator
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *
from models.rendering import render_rays_ref, render_rays_sos, render_rays_color
# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict, EdgeEnhanceLoss

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


from pdb import set_trace as st
import time

from tqdm import tqdm

import torch.nn.functional as F
class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        print(f"experiment name: {hparams.exp_name}")
        loss_type = self.hparams.loss_type
        self.loss = loss_dict[loss_type](coef=1)
        
        self.loss_tv = loss_dict['tv_loss']() if hparams.use_tv_loss else None
        
        self.train_stage = hparams.train_stage
        self.patch_size = hparams.patch_size
        
        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.exp_name = hparams.exp_name
        self.loss_ce = None
        self.color_class_module = None
        self.color_hist = None
        self.loss_edge = None
        

        self.models = {}
        if self.train_stage == 2:
            
            self.use_color_class_loss = hparams.use_color_class_loss
            if hparams.use_color_class_loss:
                print('Using color classification loss, in stage 2, after ', 
                    hparams.color_class_start_epoch, ' epochs.')
                # self.loss_ce = torch.nn.CrossEntropyLoss()
                self.loss_ce = loss_dict['ce2d']()
                print('Color temperature: ', hparams.color_class_T)
                from models.color_cls import ColorClassify
                self.color_class_module = ColorClassify(class_rebal_lambda=hparams.class_rebal_lambda, 
                                                        T=hparams.color_class_T)

            if hparams.use_edge_loss:
                print('Using edge loss, in stage 2, after ', 
                    hparams.edge_start_epoch, ' epochs.')
                self.loss_edge = EdgeEnhanceLoss(coef=1)
                
            if hparams.use_color_hist:
                # st()
                from utils.color_hist import ColorHist
                self.color_hist = ColorHist(thres=hparams.color_hist_thres, 
                                            regitser_num_max=hparams.num_color_hist, 
                                            force_accept=hparams.color_hist_force_accept)
                print(f'--Using color histogram module')
                print(f'-Threshold: {hparams.color_hist_thres}')
                print(f'-Register_num_max: {hparams.num_color_hist}')
                print(f'-Force accept: {hparams.color_hist_force_accept}')
                
                if hparams.manual_color_hist_register != 'null':
                    img_path = hparams.manual_color_hist_register
                    self.color_hist.manual_register(img_path, hparams.exp_name)
                    self.color_hist.regitser_num_max = 1
                
            assert hparams.weight_path, 'Please specify the weight path for stage 2 training'
            # assert not (hparams.use_ct2 and hparams.use_lcoder), \
            # 'Please specify the use_ct2 or use_lcoder, not both'
            self.teacher_model_name = hparams.teacher_model
            print("Using teacher model: ", self.teacher_model_name)
            if self.teacher_model_name == 'ct2': # FOR LOCAL DEBUG: NOT LOAD CT2 WEIGHTS
                # ct2 models
                from models.ct2 import CT2Wrapper
                from models.color_cls import ColorClassify
                if hparams.use_fintune_ct2:
                    model_name = hparams.fintune_ct2_name
                    print("Use finetuned CT2 model in, ", model_name)
                else:
                    model_name = None
                # print("Using CT2 as teacher model")
                self.teacher_model = CT2Wrapper(self.device, 
                                                model_name=model_name)
                
            elif self.teacher_model_name == 'lcoder':
                # L-coder models
                from models.lcoder import LCoderWrapper
                model_name = None
                # print("Using L-Coder as teacher model")
                self.teacher_model = LCoderWrapper(device=self.device, 
                                                   cap=hparams.lcoder_caption,
                                                   model_name=model_name)
                
            elif self.teacher_model_name == 'zhang':
                # zhang models
                from models.zhang import ZhangColorWrapper
                # print(self.device)
                self.teacher_model = ZhangColorWrapper(self.device, mode='fix')

        out_channels = 313 if hparams.use_color_class_loss else 2
        self.nerf_coarse = NeRF_COLOR(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                    in_channels_dir=6*hparams.N_emb_dir+3, 
                                    train_stage=self.train_stage, 
                                    use_scene_code=hparams.use_scene_code, 
                                    dense=hparams.dense, 
                                    out_channels=out_channels,
                                    )
        
        self.models['coarse'] = self.nerf_coarse
        
        words_to_ignore=['distill'] if self.train_stage == 2 else []
        load_ckpt(self.nerf_coarse, 
                  hparams.weight_path, 
                  'nerf_coarse', 
                  words_to_ignore=words_to_ignore)
        
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF_COLOR(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                      in_channels_dir=6*hparams.N_emb_dir+3, 
                                      train_stage=self.train_stage,  
                                      use_scene_code=hparams.use_scene_code, 
                                      dense=hparams.dense, 
                                      out_channels=out_channels, 
                                      )
            
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, 
                      hparams.weight_path, 
                      'nerf_fine', 
                      words_to_ignore=words_to_ignore)
        
    def setup(self, stage):
        self.logger.log_hyperparams(self.hparams)
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh), 
                  'read_gray': self.hparams.read_gray,
                  'ffr_dir': self.hparams.ffr_dir,
                  'use_patch': self.hparams.use_patch,
                  'patch_size':self.hparams.patch_size,
                  'use_coarse_rgb': self.hparams.use_coarse_rgb,
                  'use_ref': self.hparams.use_ref,
                  'patch_sample_method': self.hparams.patch_sample_method,
                  'normalize_illu': self.hparams.normalize_illu,
                  'local_image_read': False,
                  'val_num': 2,
                  'create_pose_method': self.hparams.create_pose_method,
                  }
        if 'llff' in self.hparams.dataset_name:
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)
        if self.train_stage == 2 and \
            self.hparams.zhang_mode == 'fix' and \
            self.hparams.teacher_model == 'zhang':
            
            ffr_wrapper, img_paths = self.train_dataset.get_ffr_wrapper()
            reg_0, hint_0 = self.teacher_model.register_imgset(ffr_wrapper, 
                                               img_paths, 
                                               tuple(self.hparams.img_wh), 
                                               verbose=False)
            reg_0 = reg_0[0]
            hint_0 = hint_0[0]
            stack = torch.stack([reg_0, hint_0], dim=0)
            self.logger.experiment.add_images('train/teacher_reg_hint',
                                                  stack, 0)
            
            
        if self.hparams.use_scene_code:
            N_imgs = self.train_dataset.N_imgs
            self.models['coarse'].set_N_imgs(N_imgs)
            self.models['fine'].set_N_imgs(N_imgs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        real_batch_size = 1 if self.hparams.use_patch else self.hparams.batch_size 
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=real_batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
            
    def forward(self, rays, idx=None):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        # for i in tqdm(range(0, B, self.hparams.chunk)):
        for i in range(0, B, self.hparams.chunk):
            
            # ticks = time.time()
            rendered_ray_chunks = \
                render_rays_color(models=self.models,
                            embeddings=self.embeddings,
                            rays=rays[i:i+self.hparams.chunk],
                            N_samples=self.hparams.N_samples,
                            use_disp=self.hparams.use_disp,
                            perturb=self.hparams.perturb,
                            noise_std=self.hparams.noise_std,
                            N_importance=self.hparams.N_importance,
                            chunk=self.hparams.chunk, # chunk size is effective in val mode
                            white_black=self.train_dataset.white_back,
                            train_stage=self.train_stage,
                            img_idx=idx # idx is consistent within a batch
                            )
            # tok1 = time.time()
            # inference ab results inplace
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
                if self.train_stage == 2:
                    key_ls = ['distill_rgb_fine', 'distill_rgb_coarse']
                    if k in key_ls:
                        if self.color_class_module is not None:
                            # reshape to have 4 channels
                            # v: [B, 313] in any range, 
                            #   but self.color_class_module.get_ab_infer have softmax
                            in_img = v[:, :, None, None]
                            # in_img = results[f'distill_rgb_fine'][:, :, None, None]
                            # self.color_class_module.get_ab_infer(in_img)
                            ab_img = self.color_class_module.get_ab_infer(in_img)
                        else:
                            # v: [B, 313] in range [0, 1]
                            ab_img = v[:, :, None, None] * 255 - 128
                        # st()
                        results[k+'_ab'] += [ab_img[:, :2, 0, 0]] # -128~127
            # tok2 = time.time()
            # print(f'Forward time: {tok1-ticks}')
            # print(f'Post processing time: {tok2-tok1}')
            # Forward time: 0.6487743854522705
            # Post processing time: 0.002423524856567383
                
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        
        # inference ab results on site
        # OOM with the following code...
        # if self.train_stage == 2:
        #     key_ls = ['distill_rgb_fine', 'distill_rgb_coarse']
        #     for key in key_ls:
        #         # reshape to have 4 channels
        #         in_img = results[key][:, :, None, None]
        #         ab_img = self.color_class_module.encode_ab(in_img)
        #         results[key+'_ab'] = ab_img[:, :2, 0, 0]
        # st()
        return results


    def _convert_to_gray(self, rgbs):
        grays = 0.299 * rgbs[:, 0] + 0.587 * rgbs[:, 1] + 0.114 * rgbs[:, 2]
        grays = grays.unsqueeze(1)
        grays = grays.repeat(1, 3)
        return grays 

    def training_step(self, batch, batch_nb):
        rays, rgbs, idx = batch['rays'], batch['rgbs'], batch['idx']
        # cv2.imwrite('verbose/patch_rgb.png', 255.0 * rgbs.cpu().numpy().reshape(64, 64, 3))
        if self.hparams.use_patch:
            # remove the added batch dimension
            rays = rays.squeeze(0)
            rgbs = rgbs.squeeze(0)
        
        results = self(rays, idx)
        
        self.loss_flag = True # when could not compute loss, set to False, and loss will be 0

        if self.train_stage == 1:
            prefix = 'rgb'
            grays = batch['grays']
            loss = self.loss(results, grays, prefix=prefix)
            
        elif self.train_stage == 2:
            prefix = 'distill_rgb'
            p_gray = results['rgb_fine'].view(self.patch_size, 
                                              self.patch_size, 3)
            gray_patch_orig_ = p_gray.permute(2, 0, 1).unsqueeze(0)
            gray_patch_ = F.interpolate(gray_patch_orig_,
                                        size=(256, 256), 
                                        mode='bilinear')
            gray_patch_ = gray_patch_.detach()
            
            pred_flat = self.model_inference(results)

            if self.teacher_model_name != 'zhang':
                teacher_color = self.teacher_model.inference(gray_patch_)
            else:
                # zhang model extract color hints from rgb model.
                rgb_patch = rgbs.view(self.patch_size, 
                                      self.patch_size, 
                                      3).permute(2, 0, 1).unsqueeze(0)
                input_list = [rgb_patch, batch['inds'], idx]
                # st()
                teacher_color, hint = self.teacher_model.inference(input_list, 
                                                                   verbose=True)

            
            teacher_color_patch = F.interpolate(teacher_color, 
                                                size=(self.patch_size, 
                                                      self.patch_size), 
                                                mode='bilinear')
            teacher_color_patch = torch.clamp(teacher_color_patch, 0, 1)

            teacher_color_patch = teacher_color_patch.squeeze(0).permute(1, 2, 0).detach()
            teacher_color_patch_flat = teacher_color_patch.view(-1, 3)
            # st()
            
            if torch.isnan(teacher_color_patch_flat).any():
                print("teacher_color_patch_flat has nan, skipping batch...")
                # Pytorch Lightning does NOT support returning None in training_step during ddp
                # teacher_color_patch_flat = results['distill_rgb_fine'].detach()
                teacher_color_patch_flat = pred_flat
                self.loss_flag = False
            
            if self.color_hist is not None:
                if len(self.color_hist.src_img_ls) < self.color_hist.regitser_num_max:
                    save_idx, total_idx = self.color_hist.reigister_source_img(teacher_color_patch, self.exp_name)
                    # save image to tensorboard
                    if save_idx == total_idx:
                        stack = torch.cat(self.color_hist.src_img_tensor_ls, dim=0)
                        self.logger.experiment.add_images('color_hist_src', stack)
                else:
                    color_hist_sim, score = self.color_hist.color_similarity(teacher_color_patch, 
                                                                             self.hparams.color_hist_verbose)
                    # print("score: ", score)
                    self.log("train/color_hist_score", score, prog_bar=True, logger=True)
                    if not color_hist_sim:
                        # verbose
                        if self.hparams.color_hist_verbose:
                            print('color hist not match, with score {:.5f}. skipping batch...'.format(score))
                            print(f'Total match ratio: {self.color_hist.get_match_ratio()}')
                        
                        # teacher_color_patch_flat = results['distill_rgb_fine'].detach()
                        teacher_color_patch_flat = pred_flat
                        self.loss_flag = False
            
            teacher_lab = rgb2lab(teacher_color_patch_flat)
            # st()
            loss = 0
            
            loss_l2 = self.loss(results, 
                                teacher_lab[:, 1:], 
                                prefix=prefix, 
                                suffix='_ab', 
                                normalize=True)
            
            self.log('l_l2', loss_l2)
            loss += self.hparams.weight_l2_loss * loss_l2
            
            # st()
            if self.loss_tv is not None:
                pred_img = pred_flat.view(self.patch_size, self.patch_size, 3)
                pred_img = pred_img.permute(2, 0, 1).unsqueeze(0)
                loss_tv = self.loss_tv(pred_img)
                # self.log('l_tv', loss_tv, prog_bar=True)
                loss += self.hparams.weight_tv_loss * loss_tv
                # st()

            if self.color_class_module is not None:
                if self.current_epoch >= self.hparams.color_class_start_epoch:
                    if self.current_epoch == self.hparams.color_class_start_epoch:
                        if batch_nb == 0:
                            print('Start using color classification module')
                        self.start_color_cls = True

                    gt = teacher_color_patch.unsqueeze(0).permute(0, 3, 1, 2)
                    q_gt = self.color_class_module.get_q(gt)
                    
                    loss_ce = 0
                    for suffix in ['coarse', 'fine']:
                        _key = f'{prefix}_{suffix}'
                        _p = results[_key].view(self.patch_size, self.patch_size, 313)
                        q_pred = _p.permute(2, 0, 1).unsqueeze(0)
                        q_pred_weighted = self.color_class_module.get_weighted_q(q_pred, q_gt)
                        
                        loss_ce += self.loss_ce(q_pred_weighted, q_gt)

                    self.log('l_ce', loss_ce, prog_bar=True)
                    loss += self.hparams.weight_color_class * loss_ce
            
            if self.loss_edge is not None:
                # deprecated
                if self.current_epoch >= self.hparams.edge_start_epoch:
                    if self.current_epoch == self.hparams.edge_start_epoch:
                        if batch_nb == 0:
                            print('Start using edge loss')
                        self.start_edge = True

                    loss_edge = 0
                    gt = teacher_color_patch.unsqueeze(0).permute(0, 3, 1, 2)
                    for suffix in ['coarse', 'fine']:
                        _key = f'{prefix}_{suffix}'
                        _p = results[_key].view(self.patch_size, 
                                                self.patch_size, 3)
                        pred = _p.permute(2, 0, 1).unsqueeze(0)
                        loss_edge += self.loss_edge(pred, gt)
                    
                    self.log('l_edge', loss_edge, prog_bar=True)
                    loss += self.hparams.weight_edge_loss * loss_edge
            
            
        with torch.no_grad():
            if self.train_stage == 1:
                typ = 'fine' if f'{prefix}_fine' in results else 'coarse'
                psnr_ = psnr_tensor(results[f'{prefix}_{typ}'], rgbs)
            else:
                typ = 'fine' if f'{prefix}_fine' in results else 'coarse'
                pred_flat = pred_flat
                # psnr_ = psnr_tensor(results[f'{prefix}_{typ}'], rgbs)
                psnr_ = psnr_tensor(pred_flat, rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/psnr', psnr_, prog_bar=True)
        self.log('train/loss', loss, prog_bar=True)
        if batch_nb % 50 == 0:
            if self.train_stage == 2:
                prefix = 'distill_rgb'
                W, H = self.patch_size, self.patch_size
                img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
                # img_pred = results[f'{prefix}_fine'].view(H, W, 3).permute(2, 0, 1).cpu()
                img_pred = pred_flat.view(H, W, 3).permute(2, 0, 1).cpu()
                img_teacher = teacher_color_patch.permute(2, 0, 1).cpu()
                # st()
                results[f'distill_rgb_distill_ab']
                
                img_gray_input = gray_patch_orig_.squeeze(0).cpu()
                depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
                stack = torch.stack([img_gt, 
                                     img_pred, 
                                     img_teacher,
                                     img_gray_input, 
                                     depth], dim=0)
                self.logger.experiment.add_images('train/GT_pred_teacher_gray_depth',
                                                  stack, self.global_step)
                
        if self.loss_flag == False:
            if hparams.num_gpus == 1:
                loss = None
                print('skipping batch...')
            else:
                print('soft skipping batch...')
            # print('skipping batch...')
            

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs, idx = batch['rays'], batch['rgbs'], batch['idx']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays, idx)
        # ref = self.train_dataset.ref_data['rgb']
        grays = self._convert_to_gray(rgbs)
        
        if self.train_stage == 1:
            prefix = 'rgb'
            suffix = ''
            loss = self.loss(results, grays, prefix=prefix)
        elif self.train_stage == 2:
            prefix = 'distill_rgb'
            suffix = '_ab'
            labs = rgb2lab(rgbs)
            loss = self.loss(results, labs[:, 1:], 
                             prefix=prefix, 
                             suffix=suffix, 
                             normalize=True)

        log = {'val_loss': loss}
        
        typ = 'fine' if f'{prefix}_fine' in results else 'coarse'
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            # expand to classic funcition.
            if self.train_stage == 2:
                img_flat = self.model_inference(results)
                img_flat = img_flat.cpu()
                img = img_flat.view(H, W, 3).permute(2, 0, 1)
            else:
                img = results[f'{prefix}_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                
                img_flat = img.permute(1, 2, 0).view(-1, 3)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            # st()
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        psnr_ = psnr_tensor(img_flat, rgbs.cpu())
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)

    def model_inference(self, results):
        """Model inference using L branch and ab branch of the model.
            only used in training stage 2, on _ab suffix.

        Args:
            results (dict): results dict
        Returns:
            image: inferenced RGB image [B, 3]
        """
        prefix = 'distill_rgb'
        typ = 'fine' if f'{prefix}_fine' in results else 'coarse'
        
        img_l = results[f'rgb_{typ}'].view(-1, 3) # (B, 3)
        img_l = img_l[:, 0] # (B)
        img_l = img_l[:, None] # (B, 1)
        img_l = (img_l * 100).clamp(0, 100) # (B, 1)
        img_ab = results[f'distill_rgb_{typ}_ab'].view(-1, 2) # (B, 2)
        
        img_lab = torch.cat([img_l, img_ab], dim=1) # (B, 3)
        
        img_rgb = lab2rgb(img_lab[:, :, None, None]) # (B, 3, 1, 1)
        img_rgb_flat = img_rgb.view(-1, 3) # (B, 3)
        return img_rgb_flat


def main(hparams):
    system = NeRFSystem(hparams)
    # Local environment
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                                filename='{epoch:d}',
                                monitor='val/psnr',
                                mode='max',
                                save_last=True,
                                save_top_k=2)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]
    
    os.makedirs('logs_rebuttal', exist_ok=True)
    logger = TensorBoardLogger(save_dir="logs_rebuttal",
                                name=hparams.exp_name,
                                default_hp_metric=False)
    
    trainer = Trainer(max_epochs=hparams.num_epochs,
                        callbacks=callbacks,
                        resume_from_checkpoint=hparams.ckpt_path,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        num_sanity_val_steps=hparams.val_sanity_epoch,
                        benchmark=True,
                        profiler=None,
                        strategy='ddp' if hparams.num_gpus > 1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
