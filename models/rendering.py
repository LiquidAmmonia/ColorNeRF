import torch
from einops import rearrange, reduce, repeat

import torch.functional as F
from kornia import create_meshgrid

from pdb import set_trace as st
import time

__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: (N_rays, N_importance) the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]

            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                            # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk]], 1)
                out_chunks += [model(xyzdir_embedded, sigma_only=False)]

            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=4)
            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            sigmas = out[..., 3] # (N_rays, N_samples_)
            
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse' and 'fine' in models:
            return

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, **kwargs)

    return results

def render_rays_color(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                train_stage=1,
                img_idx=None,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, 
                  test_time=False, 
                  train_stage=1, 
                  **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        # st()
        key_prefix = '' if train_stage == 1 else 'distill_'
        
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ == 'coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                
                if train_stage == 1:
                    out_chunks += [model.rgb_forward(xyz_embedded, 
                                                     sigma_only=True)]
                elif train_stage == 2:
                    out_chunks += [model.distill_forward(xyz_embedded, 
                                                         sigma_only=True, 
                                                         img_idx=img_idx)]
                    
            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                            # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk]], 1)
                # st()
                if train_stage == 1:
                    num_color_channel = 3
                    out_chunks += [model.rgb_forward(xyzdir_embedded, sigma_only=False)]
                elif train_stage == 2:
                    num_color_channel = model.out_channels # could be 313 or 2
                    out_chunks += [model.distill_forward(xyzdir_embedded, 
                                                         sigma_only=False, 
                                                         img_idx=img_idx)]
            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            # distribution color out : (N_rays, N_samples_, 313 + 1)
            out = rearrange(out, 
                            '(n1 n2) c -> n1 n2 c', 
                            n1=N_rays, 
                            n2=N_samples_, 
                            c=num_color_channel + 1)
            
            rgbs = out[..., :num_color_channel] # (N_rays, N_samples_, 3)
            sigmas = out[..., num_color_channel] # (N_rays, N_samples_)
        
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1 - torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)


        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1) # [1, 1-a1, 1-a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically


        results[f'{key_prefix}weights_{typ}'] = weights
        results[f'{key_prefix}opacity_{typ}'] = weights_sum
        results[f'{key_prefix}z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse' and 'fine' in models:
            return

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * rgbs,
                         'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'{key_prefix}rgb_{typ}'] = rgb_map
        results[f'{key_prefix}depth_{typ}'] = depth_map

        return

    # toks = []
    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']
    # toks.append(time.time())
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    z_vals_orig = z_vals.clone()
    
    # toks.append(time.time())
    
    results = {}
    # stage 1 using rgb branches
    # also needed for stage 2, because we need grayscale images.
    inference(results, models['coarse'], 'coarse', xyz_coarse, 
              z_vals, test_time, train_stage=1, **kwargs)
    # inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, train_stage=1, **kwargs)

    # toks.append(time.time())
    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference(results, models['fine'], 'fine', xyz_fine, 
                  z_vals, test_time, train_stage=1, **kwargs)
        # inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, train_stage=1, **kwargs)
    # toks.append(time.time())
    # print(toks)
    # st()
    if train_stage == 2:
        # print('stage 2')
        z_vals = z_vals_orig
        inference(results, models['coarse'], 'coarse', xyz_coarse, 
                z_vals, test_time, train_stage=train_stage, **kwargs)

        if N_importance > 0: # sample points for fine model
            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
            z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                                N_importance, det=(perturb==0))
                    # detach so that grad doesn't propogate to weights_coarse from here

            z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                    # combine coarse and fine samples
            
            xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

            inference(results, models['fine'], 'fine', xyz_fine, 
                    z_vals, test_time, train_stage=train_stage, **kwargs)

    return results


def render_rays_sos(models,
                    embeddings,
                    rays,
                    N_samples=64,
                    use_disp=False,
                    perturb=0,
                    noise_std=1,
                    N_importance=0,
                    chunk=1024*32,
                    white_back=False,
                    test_time=False,
                    train_stage=1,
                    img_idx=None,
                    **kwargs
                    ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, 
                  test_time=False, 
                  train_stage=1, 
                  **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        key_prefix = '' if train_stage == 1 else 'distill_'
        
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ == 'coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                if train_stage == 1:
                    out_chunks += [model.rgb_forward(xyz_embedded, 
                                                     sigma_only=True)]
                elif train_stage == 2:
                    out_chunks += [model.distill_forward(xyz_embedded, 
                                                         sigma_only=True, 
                                                         img_idx=img_idx)]
                    
            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                            # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk]], 1)
                # st()
                if train_stage == 1:
                    out_chunks += [model.rgb_forward(xyzdir_embedded, sigma_only=False)]
                elif train_stage == 2:
                    out_chunks += [model.distill_forward(xyzdir_embedded, 
                                                         sigma_only=False, 
                                                         img_idx=img_idx)]

            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=4)
            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            sigmas = out[..., 3] # (N_rays, N_samples_)
         
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1 - torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)


        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1) # [1, 1-a1, 1-a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'{key_prefix}weights_{typ}'] = weights
        results[f'{key_prefix}opacity_{typ}'] = weights_sum
        results[f'{key_prefix}z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse' and 'fine' in models:
            return

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * rgbs,
                         'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'{key_prefix}rgb_{typ}'] = rgb_map
        results[f'{key_prefix}depth_{typ}'] = depth_map

        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    z_vals_orig = z_vals.clone()
    
    results = {}
    # stage 1 using rgb branches
    inference(results, models['coarse'], 'coarse', xyz_coarse, 
              z_vals, test_time, train_stage=1, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference(results, models['fine'], 'fine', xyz_fine, 
                  z_vals, test_time, train_stage=1, **kwargs)

    if train_stage == 2:
        # print('stage 2')
        z_vals = z_vals_orig
        inference(results, models['coarse'], 'coarse', xyz_coarse, 
                z_vals, test_time, train_stage=train_stage, **kwargs)

        if N_importance > 0: # sample points for fine model
            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
            z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                                N_importance, det=(perturb==0))
                    # detach so that grad doesn't propogate to weights_coarse from here

            z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                    # combine coarse and fine samples
            
            xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

            inference(results, models['fine'], 'fine', xyz_fine, 
                    z_vals, test_time, train_stage=train_stage, **kwargs)

    return results


def render_rays_ref(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                ref_data=None,
                img_idx=None,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, warped_fea=None, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)
        # st()
        if warped_fea is not None:
            warped_fea_ = rearrange(warped_fea, 'n1 n2 c -> (n1 n2) c')

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]

            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                            # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                if warped_fea is not None:
                    xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk], 
                                            warped_fea_[i:i+chunk],
                                             ],1)
                
                out_chunks += [model(xyzdir_embedded, sigma_only=False)]

            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=4)
            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            sigmas = out[..., 3] # (N_rays, N_samples_)
            
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse' and 'fine' in models:
            return

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        return

    def warp_features_to_ref(xyz, ref_data, near=0., far=1.0, pad=0, eps=1e-10):
        """
        Helper function that warps features from the reference view to the current view.
        Inputs:
            xyz: (N_rays, N_samples, 3) sample origins

            eps: a small number to avoid division by zero
        Outputs:
            warped_depth: (N_rays, N_samples) depth map warped from the reference view
        """
        
        device = xyz.device
        c2w_ref = ref_data['c2w'][0].to(device)
        intrinsic_ref = ref_data['intrinsic'][0].to(device)
        rgb_ref = ref_data['rgb'][0].to(device)
        H, W = rgb_ref.shape[:2]
        inv_scale = torch.tensor([W-1, H-1]).to(device)
        
        _c2w = torch.eye(4).float().to(device)
        _c2w[:3] = c2w_ref
        w2c_ref = torch.inverse(_c2w)

        # inspired from mvsnerf/utils.py [get_ndc_coordinate]
        R = w2c_ref[:3, :3]
        T = w2c_ref[:3, 3:]

        # ref_xyz = torch.matmul(xyz, R.t()) + T.reshape(1, 3)
        xyz = xyz.permute(2, 0, 1) # [3, N_rays, N_samples]
        N_rays = xyz.shape[1]
        xyz_flatten = xyz.reshape(3, -1) # [3, N_rays*N_samples]
        
        ref_xyz = R @ xyz_flatten + T.reshape(3, 1)
        
        ref_xyz_pixel = intrinsic_ref @ ref_xyz
        
        # clip to right pixel region
        ref_xyz_pixel[1] = ref_xyz_pixel[1] * H / (H + pad * 2) + pad / (H + pad * 2)
        ref_xyz_pixel[0] = ref_xyz_pixel[0] * W / (W + pad * 2) + pad / (W + pad * 2)
        
        ref_xyz_pixel[1] = ref_xyz_pixel[1] * (H / 2) + (H / 2)
        ref_xyz_pixel[0] = ref_xyz_pixel[0] * (W / 2) + (W / 2)
        
        ref_h = torch.clamp(ref_xyz_pixel[1], 0, H-1).long()
        ref_w = torch.clamp(ref_xyz_pixel[0], 0, W-1).long()
        
        warped_feature = rgb_ref[ref_h, ref_w, :]
        warped_feature = warped_feature.reshape(3, N_rays, -1).permute(1, 2, 0)

        return warped_feature
        
    def homo_w(src_feat, proj_mat, depth_values, pad=0):
        """
        src_feat: (C, H, W)
        proj_mat: (3, 4) equal to "src_proj @ ref_proj_inv"
        depth_values: (H, W)
        out: (C, H, W)
        """
        
        src_feat = src_feat.permute(2, 0, 1) # [C, H, W]
        C, H, W = src_feat.shape
        device = src_feat.device
        
        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        R = proj_mat[:, :3]  # (3, 3)
        T = proj_mat[:, 3:]  # (3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (1, 3, H*W)
        # ref_grid_d = ref_grid.repeat(1, D)  # (3, D*H*W)
        
        # could not figure out the dimentionalities of depth_values, pin on that
        src_grid_d = R @ ref_grid + T / depth_values.view(1, W_pad * H_pad)
        del ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

        src_grid = src_grid_d[0,:2] / src_grid_d[0,2:]  # divide by depth (., H*W)
        del src_grid_d
        src_grid[0] = src_grid[0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[1] = src_grid[1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.T  # (B, H*W, 2)
        src_grid = src_grid.view(1, W_pad,H_pad, 2)

        B, W_pad, H_pad = src_grid.shape[:3]
        warped_src_feat = F.grid_sample(src_feat.unsqueeze(0), src_grid,
                                        mode='bilinear', padding_mode='zeros',
                                        align_corners=True)  # (C, H*W)
        warped_src_feat = warped_src_feat.view(C,H_pad,W_pad)
    

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    assert ref_data is not None, 'ref_sample should not be None'


    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    rays_coords = rays[:, 8:10] # (N_rays, 2)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    rays_coords = rearrange(rays_coords, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
    
    
    # warped_feature = warp_features_to_ref(xyz=xyz_coarse, 
    #                                     ref_data=ref_data,)
    st()
    warped_feature = homo_w(src_feat=ref_data['rgb'][0],
                            proj_mat=ref_data['all_proj_mats'][img_idx],
                            depth_values=z_vals)

    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, warped_fea=warped_feature, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
        # only add warped feature to fine model (along with in_dirs)
        warped_feature = warp_features_to_ref(xyz=xyz_fine, 
                                        ref_data=ref_data,)
                                        

        inference(results, models['fine'], 'fine', xyz_fine, z_vals, test_time, warped_fea=warped_feature, **kwargs)

    return results
