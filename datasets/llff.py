import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os,sys
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from pdb import set_trace as st
import cv2


o_path = os.getcwd()
sys.path.append(o_path)


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def create_spiral_poses(radii, focus_depth, n_poses=60):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 2*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth])) # smaller focus depth means closer 
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spike_1_poses(radii, focus_depth, n_poses=60):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 2*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)



def create_straight_poses(train_poses, radii, focus_depth, n_poses=120):
    """
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """
    orig_pose = train_poses[0]
    poses_spiral = []
    for t in np.linspace(-0.6, 0.4, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        new_pose = orig_pose.copy()
        new_t_delta = np.array([t, -t, 0]) * radii
        # st()
        new_pose[:, 3] += new_t_delta
        poses_spiral += [new_pose]
    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378), spheric_poses=False, val_num=1, ref_idx=16, read_gray=False, use_patch=False, use_coarse_rgb=False, use_ref=False, patch_sample_method='central', normalize_illu=False, patch_size=64, local_image_read=False, create_pose_method='spheric', orig_img=False, focus_depth=3.5, **kwargs):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        ref_idx: index of reference image
        """
        # initialize variables
        self.focus_depth = focus_depth
        self.local_full_dir = '/userhome/chengyean/nerf_llff_data_images'
        self.local_bins_dir = './nerf_llff_data_bins'
        self.scene_name = os.path.basename(root_dir)
        # st()
        # self.root_dir = os.path.join(self.local_bins_dir, self.scene_name)
        self.root_dir = os.path.join(self.local_full_dir, self.scene_name)

        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.ref_idx = -1
        
        self.ref_data = None
        self.white_back = False
        
        self.use_patch = use_patch if split == 'train' else False
        self.use_coarse_rgb = use_coarse_rgb if split == 'train' else False
        
        self.patch_sample_method = patch_sample_method
        self.patch_size = patch_size
        
        self.normalize_illu = normalize_illu
        self.local_image_read = local_image_read
        self.create_pose_method = create_pose_method
        
        self.orig_img = orig_img
        # define transforms and read data
        self.define_transforms()

        self.read_meta()
        
        
        if self.use_patch:
            self.patch_size = self.patch_size
            print('Using batching, Patch size: ', self.patch_size)
            self.rays_batch_size = self.patch_size ** 2
            self._get_structured_rays()
            

    def _read_image_gray(self, path, read_gray=False):
        if not read_gray:
            # img = Image.open(path).convert('RGB')
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # expand to 3 channels
            img = np.stack([img, img, img], 2)
            
        return img

    
    def read_meta(self):
        # Step 1: rescale focal length according to training resolution
        
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        self.focal = camdata[1].params[0] * self.img_wh[0]/W

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])
        # read successfully reconstructed images and ignore others
        # self.image_paths = [os.path.join(self.root_dir, 'images', name)
        #                     for name in sorted([imdata[k].name for k in imdata])]
        
        self.image_paths = [os.path.join(self.scene_name, 'images', name)
                            for name in sorted([imdata[k].name for k in imdata])]
        
        # hard coded for coarse rgb images
        if self.use_coarse_rgb:
            self.coarse_rgb_root = './utils/llff_16'
            print("Using coarse rgb images from ", self.coarse_rgb_root)
            self.coarse_rgb_paths = [os.path.join(self.coarse_rgb_root, 
                                                name.replace('.JPG', '_W_16.JPG'))
                                for name in sorted([imdata[k].name for k in imdata])]
            
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices
        
        # read bounds
        self.bounds = np.zeros((len(poses), 2)) # (N_images, 2)
        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts_world = np.zeros((1, 3, len(pts3d))) # (1, 3, N_points)
        visibilities = np.zeros((len(poses), len(pts3d))) # (N_images, N_points)
        
        for i, k in enumerate(pts3d):
            pts_world[0, :, i] = pts3d[k].xyz
            for j in pts3d[k].image_ids:
                visibilities[j-1, i] = 1
        # calculate each point's depth w.r.t. each camera
        # it's the dot product of "points - camera center" and "camera frontal axis"
        depths = ((pts_world-poses[..., 3:4])*poses[..., 2:3]).sum(1) # (N_images, N_points)
        for i in range(len(poses)):
            visibility_i = visibilities[i]
            zs = depths[i][visibility_i==1]
            self.bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]
        # permute the matrices to increasing order
        poses = poses[perm]
        self.bounds = self.bounds[perm]
        
        # COLMAP poses has rotation in form "right down front", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses, _ = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image
        self.val_idx = val_idx

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
        
        # calculate the grayscale img mean as illumination
        illum_ls = []
        
        for i, image_path in enumerate(self.image_paths):
            if not self.local_image_read:
                img_path = os.path.join(self.local_full_dir, image_path)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_path = os.path.join(self.local_bins_dir, image_path)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # st()
            img = cv2.resize(np.array(img), self.img_wh, interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            illum = gray.mean()
            illum_ls.append(illum)
        self.illum_mean = np.mean(illum_ls)
        print("Grayscale mean: ", self.illum_mean)
        # st()
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            self.all_coarse_rgbs = []
            self.all_c2ws = []

            for i, image_path in enumerate(self.image_paths):
                if i == val_idx: # exclude the val image
                    continue
                if i == self.ref_idx:
                    print("Reference image: ", image_path)
                    continue
                
                c2w = torch.FloatTensor(self.poses[i])
                
                if not self.local_image_read:
                    img_path = os.path.join(self.local_full_dir, image_path)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_path = os.path.join(self.local_bins_dir, image_path)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # st()
                img = cv2.resize(np.array(img), self.img_wh, interpolation=cv2.INTER_LANCZOS4)
                img = self.transform(img) # (3, h, w)
                # st()
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                if i != self.ref_idx:
                    self.all_rgbs += [img]
                
                if self.use_coarse_rgb:
                    coarse_rgb_path = self.coarse_rgb_paths[i]
                    coarse_rgb = cv2.imread(coarse_rgb_path, cv2.IMREAD_UNCHANGED)
                    coarse_rgb = cv2.cvtColor(coarse_rgb, cv2.COLOR_BGR2RGB)
                    coarse_rgb = cv2.resize(np.array(coarse_rgb), 
                                            self.img_wh, 
                                            interpolation=cv2.INTER_LANCZOS4)
                    coarse_rgb = self.transform(coarse_rgb) # (3, h, w)
                    coarse_rgb = coarse_rgb.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    if i != self.ref_idx:
                        self.all_coarse_rgbs += [coarse_rgb]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                # rays_o, rays_d, rays_coord = get_rays_mvs(self.directions, c2w)
                
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only
                
                cat_rays = torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1]),
                                             ],1) # (h*w, 8+2)
                if i != self.ref_idx:
                    self.all_rays += [cat_rays]
                    
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 10)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            if self.use_coarse_rgb:
                self.all_coarse_rgbs = torch.cat(self.all_coarse_rgbs, 0) # ((N_images-1)*h*w, 3)
            
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.val_idx = val_idx
        
        elif self.split == 'evaluate':
            print('val image is', self.image_paths[val_idx])
            self.val_idx = val_idx
        
        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                # focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                # given in the original repo. Mathematically if near=1
                                # and far=infinity, then this number will converge to 4
                focus_depth = self.focus_depth
                print("Focus depth is ", focus_depth, "m")
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                if self.create_pose_method == 'spheric':
                    self.poses_test = create_spiral_poses(radii, focus_depth)
                if self.create_pose_method == 'nir':
                    focus_depth = 1000
                    self.poses_test = create_spiral_poses(radii, focus_depth)
                if self.create_pose_method == 'movie2':
                    focus_depth = 10
                    self.poses_test = create_spiral_poses(radii, focus_depth)
                elif self.create_pose_method == 'straight':
                    self.poses_test = create_straight_poses(self.poses, 
                                                            radii, 
                                                            focus_depth, 
                                                            n_poses=60)
                elif self.create_pose_method == 'spike_1':
                    self.poses_test = create_spiral_poses(radii * 0.3, focus_depth)
                    
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)
                

            
    def _get_structured_rays(self):
        """
        Convert self.rays and self.rgbs to [N_img, H, W, 3] for use_patch. 
        Only used for training.
        """
        assert self.split == 'train'
        H, W = self.img_wh[1], self.img_wh[0]

        N_imgs = int(self.all_rgbs.shape[0] / (H*W))
        print("Getting Structured Rays. N_imgs: ", N_imgs)
        self.s_all_rays = self.all_rays.reshape(N_imgs, H, W, -1)
        self.s_all_rgbs = self.all_rgbs.reshape(N_imgs, H, W, -1)
        if self.use_coarse_rgb:
            self.s_all_coarse_rgbs = self.all_coarse_rgbs.reshape(N_imgs, H, W, -1)
        
        self.len_multiplier = int(H * W / (self.patch_size ** 2))
        print("len_multiplier: ", self.len_multiplier)
        self.N_imgs = N_imgs
   
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            if not self.use_patch:
                return len(self.all_rays)
            else:
                return len(self.s_all_rays) * self.len_multiplier
        if self.split == 'val':
            return self.val_num
        if self.split == 'evaluate':
            return self.val_num
        if self.split == 'test_train':
            return len(self.poses)
        return len(self.poses_test)

    def _get_select_inds(self, N_samples_sqrt):
        
        """
        method: 
            'central':
            'random': 
        """
        orig_w, orig_h = torch.meshgrid([torch.linspace(-1, 1, N_samples_sqrt),
                               torch.linspace(-1, 1, N_samples_sqrt)])
        h = orig_h.unsqueeze(2)
        w = orig_w.unsqueeze(2)
        
        method = self.patch_sample_method
        if method == 'random':
            min_scale = 0.3
            max_scale = 1.0
        elif method == 'central':
            min_scale = 0.5
            max_scale = 1.0
        elif method == 'full':
            min_scale = 1.0
            max_scale = 1.0
            
        # random_scale:
        scale = torch.Tensor(1).uniform_(min_scale, max_scale)
        h = h * scale
        w = w * scale
        
        # random_shift:
        max_offset = 1 - scale.item()
        h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2
        w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2
        h += h_offset
        w += w_offset
            
        return torch.cat([h, w], dim=2)
    
    def _rgb_to_gray(self, rgb):
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        gray = gray.unsqueeze(2)
        gray = gray.repeat(1, 1, 3)
        return gray
    
    def _rgb_to_gray_flat(self, rgb_flat):
        gray = 0.2989 * rgb_flat[0] + 0.5870 * rgb_flat[1] + 0.1140 * rgb_flat[2]
        gray = gray.unsqueeze(0)
        gray = gray.repeat(3)
        return gray

    def _patch_sample(self, idx):
        
        s_rgb = self.s_all_rgbs[idx]
        s_ray = self.s_all_rays[idx]
        s_gray = self._rgb_to_gray(s_rgb)
        
        if self.normalize_illu:
            s_gray = s_gray / torch.mean(s_gray[:, :, 0]) * (self.illum_mean / 255.0)
            s_gray = torch.clamp(s_gray, 0, 1)
        
        patch_size = self.patch_size
        select_inds = self._get_select_inds(patch_size)
        
        if self.orig_img:
            orig_gray = s_gray
            orig_rgb = s_rgb
        
        def _grid_sample(img, inds):
            return F.grid_sample(img.permute(2, 0, 1).unsqueeze(0),
                                 inds.unsqueeze(0), 
                                 align_corners=True).squeeze(0).permute(1, 2, 0)
        
        def _flatten(img):
            return img.view(-1, img.shape[-1])
        
        s_rgb = _flatten(_grid_sample(s_rgb, select_inds))
        s_ray = _flatten(_grid_sample(s_ray, select_inds))
        s_gray = _flatten(_grid_sample(s_gray, select_inds))
        
        sample = {'rays': s_ray, 'rgbs': s_rgb, 'idx': idx, 'grays': s_gray, 
                  'inds': select_inds}
        
        if self.orig_img:
            sample['orig_gray'] = orig_gray
            sample['orig_rgb'] = orig_rgb
        
        if self.use_coarse_rgb:
            s_coarse_rgb = self.s_all_coarse_rgbs[idx]
            s_coarse_rgb = _flatten(_grid_sample(s_coarse_rgb, select_inds))
            sample['coarse_rgb'] = s_coarse_rgb
        return sample

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            if not self.use_patch:
                sample = {'rays': self.all_rays[idx],
                        'rgbs': self.all_rgbs[idx], 
                        'idx': idx, 
                        'grays': self._rgb_to_gray_flat(self.all_rgbs[idx]),
                        }
            else:
                # sample more imgs because of the patch sampling
                norm_idx = idx % self.N_imgs
                assert norm_idx < len(self.image_paths)
                sample = self._patch_sample(norm_idx)
                # st()
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            elif self.split == 'evaluate':
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            elif self.split == 'test_train':
                c2w = torch.FloatTensor(self.poses[idx])
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w, 
                      'idx': idx}
            if self.split in ['val', 'test_train', 'evaluate']:
                if self.split == 'val':
                    idx = self.val_idx
                if self.split == 'evaluate':
                    idx = self.val_idx
                if not self.local_image_read:
                    image_path = self.image_paths[idx]
                    img_path = os.path.join(self.local_full_dir, image_path)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_path = os.path.join(self.local_bins_dir, self.image_paths[idx])
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(np.array(img), self.img_wh, 
                                 interpolation=cv2.INTER_LANCZOS4)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img
        
        return sample


if __name__ == '__main__':
    root_dir = '/userhome/chengyean/ARF-svox2/data/llff/flower'
    dataset = LLFFDataset(root_dir=root_dir)
    print(len(dataset))
    
    # test selected inds
    