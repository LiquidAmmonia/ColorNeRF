
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import time

from models.rendering import render_rays, render_rays_ref, render_rays_sos, render_rays_color
from models.nerf import *
import imageio

from argparse import ArgumentParser

import metrics
from utils import load_ckpt

from datasets import dataset_dict
from datasets.depth_utils import *
from models.color_cls import ColorClassify
import os

from tqdm import tqdm
from utils.visualization import visualize_depth
import json

from utils import *
from pdb import set_trace as st


torch.backends.cudnn.benchmark = True

from eval_color import get_opts


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk, color_class_module, 
                      train_stage=1, 
                      use_color_class_loss=True):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays_color(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True, 
                        ref_data=dataset.ref_data, 
                        train_stage=train_stage)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]
            if train_stage == 2:
                key_ls = ['distill_rgb_fine', 'distill_rgb_coarse']
                if k in key_ls:
                    # reshape to have 4 channels
                    if use_color_class_loss:
                        # v: [B, 313] in any range
                        in_img = v[:, :, None, None]
                        ab_img = color_class_module.get_ab_infer(in_img)
                    else:
                        # v: [B, 2] in range [0, 1]
                        ab_img = v[:, :, None, None] * 255 - 128
                        
                    results[k+'_ab'] += [ab_img[:, :2, 0, 0]]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == '__main__':
    args = get_opts()
    w, h = args.img_wh
    torch.cuda.empty_cache()
    
    if not args.local_run:
        import hfai_env
        hfai_env.set_env('ct2_sos')
    
    kwargs = {'root_dir': args.root_dir,
              'split': 'evaluate',
              'img_wh': tuple(args.img_wh), 
              'ffr_dir': args.ffr_dir,
              }
    
    color_class_module = ColorClassify(T=args.color_class_T)
    
    if 'llff' in args.dataset_name:
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    
    out_channels = 313 if args.use_color_class_loss else 2
    def init_model(model_name, out_channels):
        
        nerf = NeRF_COLOR(in_channels_xyz=6*args.N_emb_xyz+3,
                                in_channels_dir=6*args.N_emb_dir+3,
                                use_scene_code=args.use_scene_code,
                                train_stage=args.train_stage,
                                dense=args.dense, 
                                out_channels=out_channels
                                )
        # # for scene_code
        # N_imgs = len(dataset.image_paths) - 1 # hard-coded N_imgs, must be a better way
        # nerf.set_N_imgs(N_imgs)
        load_ckpt(nerf, args.ckpt_path, model_name=model_name)
        nerf.cuda().eval()
        return nerf
    
    nerf_coarse = init_model('nerf_coarse',out_channels)
    nerf_fine = init_model('nerf_fine',out_channels)
    
    models = {'coarse': nerf_coarse, 'fine': nerf_fine}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    dir_name = f'test_results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    # metric_dict = {'psnr': [], 
    #                'ssim': [], 
    #                'lpips': []}
    metric_res = {}
    imgs, depth_maps = [], []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,color_class_module, 
                                    args.train_stage, 
                                    args.use_color_class_loss)
        
        prefix = 'rgb'
        if args.train_stage == 1:
            prefix = 'rgb'
        elif args.train_stage == 2:
            prefix = 'distill_rgb'
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        
        if args.train_stage == 2:
            img_l = results[f'rgb_{typ}'].view(h, w, 3).permute(2, 0, 1).cpu()
            img_l = img_l[0]
            # convert to [0-100] L channel
            img_l = (img_l * 100).clamp(0, 100)
            img_ab = results[f'distill_rgb_{typ}_ab'].view(h, w, 2).permute(2, 0, 1).cpu()
            img_lab = torch.cat([img_l.unsqueeze(0), img_ab], dim=0)
            
            img_pred = lab2rgb(img_lab.unsqueeze(0)).numpy() # (1, 3, H, W)
            img_pred = np.clip(img_pred[0], 0, 1)
            img_pred = img_pred.transpose(1, 2, 0)
        else:
            img_pred = np.clip(results[f'{prefix}_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)


        if args.save_depth:
            depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(os.path.join(dir_name, f'depth_{i:03d}'), 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        # imgs += [img_pred_]
        
        rgbs = sample['rgbs']
        img_gt = rgbs.view(h, w, 3).cpu().numpy()
        img_gt_ = (img_gt * 255).astype(np.uint8)
        
        gt_path = os.path.join(dir_name, 'gt')
        pred_path = os.path.join(dir_name, 'pred')
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}_pred.png'), img_pred_)
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}_gt.png'), img_gt_)
        
        
        depth_pred_ = results[f'depth_{typ}'].view(h, w)
        depth_pred = visualize_depth(depth_pred_).permute(1,2,0).numpy()
        # alpha_pred = results['opacity_fine'].view(h, w).cpu().numpy()
        # alpha_pred_dim3 = alpha_pred[:, :, None].repeat(3, axis=2)
        
        img_cat = np.concatenate([img_gt, img_pred, depth_pred], axis=1)
        img_cat_ = (img_cat * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}_full.png'), img_cat_)
        
        img_pred_tensor = torch.from_numpy(img_pred).permute(2,0,1).unsqueeze(0)
        img_gt_tensor = torch.from_numpy(img_gt).permute(2,0,1).unsqueeze(0)
        
        metric_res[f'{i:03d}'] = {}
        metric_res[f'{i:03d}']['psnr'] = str(metrics.psnr(img_pred_tensor, img_gt_tensor))
        metric_res[f'{i:03d}']['ssim'] = str(metrics.ssim(img_pred_tensor, img_gt_tensor))
        metric_res[f'{i:03d}']['lpips'] = str(metrics.lpips(img_pred_tensor.cuda(), img_gt_tensor.cuda()))
        
        # colorful metrics
        # breakpoint()
        metric_res[f'{i:03d}']['colorful'] = str(metrics.image_colorfulness(img_pred_))
        metric_res[f'{i:03d}']['delta_colorful'] = str(abs(metrics.image_colorfulness(img_pred_) -  
                                                           metrics.image_colorfulness(img_gt_)))

        # special save for fid score.
        gt_path = os.path.join(dir_name, 'gt')
        pred_path = os.path.join(dir_name, 'pred')
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(pred_path, exist_ok=True)
        imageio.imwrite(os.path.join(pred_path, f'fake_{i:03d}.png'), img_pred_)
        imageio.imwrite(os.path.join(gt_path, f'{i:03d}.png'), img_gt_)
        
        fid_score, _ = metrics.my_calculate_fid(pred_path, gt_path)
        # os.system(f'python -m {pred_path} {gt_path} --device cuda:0')
        metric_res[f'{i:03d}']['fid'] = str(fid_score)
        
        print('*' * 40)
        print(f'Image: {i:03d}') # do we need original image name?
        print(metric_res[f'{i:03d}'])
    metric_res['mean'] = {}
    
    def _calc_mean(metric_res, key):
        out_ls = [float(metric_res[k][key]) for k in metric_res.keys() if k != 'mean']
        mean = np.mean(out_ls)
        return mean
        
    metric_res['mean']['psnr'] = str(_calc_mean(metric_res, 'psnr'))
    metric_res['mean']['ssim'] = str(_calc_mean(metric_res, 'ssim'))
    metric_res['mean']['lpips'] = str(_calc_mean(metric_res, 'lpips'))
    
    # st()
    
    import pandas as pd
    # save reulsts to csv
    df = pd.DataFrame(metric_res).T
    df.to_csv(os.path.join(dir_name, 'metrics.csv'))
    
    # save metrics to json
    with open(os.path.join(dir_name, 'metrics.txt'), 'w') as f:
        f.write(json.dumps(metric_res))
    





