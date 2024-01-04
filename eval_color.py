
import os
import cv2

from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays, render_rays_ref, render_rays_sos, render_rays_color
from models.nerf import *

from utils import *
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
from models.color_cls import ColorClassify

from pdb import set_trace as st

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*16,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')
    
    parser.add_argument('--ffr_dir', type=str,
                        default='nerf_llff_data_bins/nerf_llff_data_images.ffr',
                        help='root directory of dataset')

    parser.add_argument('--train_stage', type=int, default=1,
                        help='Use semantically-aware batching')
    
    parser.add_argument('--use_scene_code', default=False, action="store_true",
                        help='Use a learnable vector to represent scene identity')

    parser.add_argument('--local_run', default=False, action="store_true",
                        help='Use a learnable vector to represent scene identity')

    parser.add_argument('--dense', default=False, action="store_true",
                        help='Use a learnable vector to represent scene identity')
    
    parser.add_argument('--use_color_class_loss', action="store_true",
                        help='Use a learnable vector to represent scene identity')
    parser.add_argument('--not_use_color_class_loss', dest='use_color_class_loss', action='store_false')
    parser.set_defaults(use_color_class_loss=True)
    
    
    parser.add_argument('--create_pose_method', type=str, default='spheric',
                        help='straight for movie(paris)')
    parser.add_argument('--color_class_T', type=float, default=0,
                        help='weight of color classification')
    parser.add_argument('--focus_depth', type=float, default=3.5,
                        help='Focus depth')
    
    
    
    parser.add_argument('--suffix', type=str, default='',
                        help='straight for movie(paris)',)

    return parser.parse_args()


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


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    if not args.local_run:
        import hfai_env
        hfai_env.set_env('ct2_sos')

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh), 
              'ffr_dir': args.ffr_dir,
              'create_pose_method': args.create_pose_method,
              'focus_depth': args.focus_depth,
              }
    
    color_class_module = ColorClassify(T=args.color_class_T)
    
    if 'llff' in args.dataset_name:
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)
    # st()
    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    # nerf_coarse = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
    #                    in_channels_dir=6*args.N_emb_dir+3)
    
    out_channels = 313 if args.use_color_class_loss else 2
    # st()
    nerf_coarse = NeRF_COLOR(in_channels_xyz=6*args.N_emb_xyz+3,
                            in_channels_dir=6*args.N_emb_dir+3,
                            use_scene_code=args.use_scene_code, 
                            train_stage=args.train_stage,
                            dense=args.dense, 
                            out_channels=out_channels
                            )
    N_imgs = len(dataset.image_paths) - 1 # hard-coded N_imgs, must be a better way
    nerf_coarse.set_N_imgs(N_imgs)
    # st()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if args.N_importance > 0:
        nerf_fine = NeRF_COLOR(in_channels_xyz=6*args.N_emb_xyz+3,
                               in_channels_dir=6*args.N_emb_dir+3, 
                               use_scene_code=args.use_scene_code,
                               train_stage=args.train_stage,
                               dense=args.dense, 
                               out_channels=out_channels,
                               )
        nerf_fine.set_N_imgs(N_imgs)
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine

    imgs, depth_maps, psnrs = [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}{args.suffix}'
    
    
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, f'{args.scene_name}_imgs'), exist_ok=True)

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

        # st()
        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{args.scene_name}_imgs', f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [0]
            # psnrs += [metrics.psnr(img_gt, img_pred).item()]

    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)

    if args.save_depth:
        min_depth = np.min(depth_maps)
        max_depth = np.max(depth_maps)
        depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
        depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_depth.gif'), depth_imgs_, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
