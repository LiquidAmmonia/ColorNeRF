import torch
from torch import nn
import time

from pdb import set_trace as st

class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, f)

        Outputs:
            out: (B, 2*f*N_freqs+f)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out
    
    def rgb_forward(self, x, sigma_only=False):
        return self.forward(x, sigma_only=sigma_only)[:, :3]


class NeRF_COLOR(nn.Module):
    def __init__(self,
                 D=8, 
                 W=256,
                 in_channels_xyz=63, 
                 in_channels_dir=27, 
                 skips=[4], 
                 train_stage=1, 
                 use_scene_code=False, 
                 distribute_color=False,
                 dense=False, 
                 out_channels=313,
                 no_view_rgb=False
                 ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        output 313 channels as the distribution of color
        train_stage: 
            1 for building grayscale nerf; 
            2 for fix the rgb and sigma mlp, just train the distill mlp
        """
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.out_channels = out_channels
        self.dense = dense
        self.train_stage = train_stage
        self.no_view_rgb = no_view_rgb

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())
        
        # add exrta rgb distill layer
        self.scene_code_dim = 128
        self.use_scene_code = use_scene_code
        
        self.distribute_color = distribute_color
        if self.use_scene_code:
            # define a learnable latent code
            # self.scene_code = nn.Parameter(torch.randn(1, self.scene_code_dim))
            self.scene_code = None
            self.scene_code_encoding = nn.Sequential(
                                        nn.Linear(self.scene_code_dim, W//2),
                                        nn.ReLU(True), 
                                        nn.Linear(W//2, W//2), 
                                        nn.ReLU(True)
                                        )
            dir_encoding_input_dim = W + in_channels_dir + W//2      
        else:
            dir_encoding_input_dim = W + in_channels_dir

        if train_stage == 2:
            if self.dense:
                self.dir_encoding_distill = nn.Sequential(
                                                nn.Linear(dir_encoding_input_dim, W),
                                                nn.ReLU(True), 
                                                nn.Linear(W, W),
                                                nn.ReLU(True), 
                                                nn.Linear(W, W),
                                                nn.ReLU(True), 
                                                )

                # self.distill = nn.Sequential(
                #                 nn.Linear(W, self.out_channels),
                #                 nn.Softmax(dim=1))
                if self.out_channels == 313:
                    self.distill = nn.Linear(W, self.out_channels)
                else:
                    self.distill = nn.Sequential(
                                nn.Linear(W, self.out_channels),
                                nn.Sigmoid())
            else:
                self.dir_encoding_distill = nn.Sequential(
                                                nn.Linear(dir_encoding_input_dim, W//2),
                                                nn.ReLU(True), 
                                                nn.Linear(W//2, W//2),
                                                nn.ReLU(True), 
                                                )

                # self.distill = nn.Sequential(
                #                 nn.Linear(W//2, self.out_channels),
                #                 nn.Softmax(dim=1))
                if self.out_channels == 313:
                    self.distill = nn.Linear(W//2, self.out_channels)
                else:
                    self.distill = nn.Sequential(
                                        nn.Linear(W//2, self.out_channels),
                                        nn.Sigmoid())
        else:
            self.dir_encoding_distill = None
            self.distill = None
            
        
        # self.distill = nn.Sequential(
        #                 nn.Linear(W//2, 3),
        #                 nn.Sigmoid())
        
        if train_stage == 2:
            print("# freeze all the parameters except self.dir_encoding_distill and self.distill")
            for param in self.parameters():
                param.requires_grad = False
            for param in self.dir_encoding_distill.parameters():
                param.requires_grad = True
            for param in self.distill.parameters():
                param.requires_grad = True
    
    def set_N_imgs(self, N_imgs):
        print("Initializing scene code with N_imgs = ", N_imgs)
        self.scene_code = nn.Embedding(N_imgs, self.scene_code_dim)

    def forward(self, x, sigma_only=False):
        return self.rgb_forward(x, sigma_only=sigma_only)
        
    def rgb_forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        # toks = []
        # toks.append(time.time())
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x
        # toks.append(time.time())
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        # toks.append(time.time())
        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma
        # toks.append(time.time())
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        # toks.append(time.time())
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        # toks.append(time.time())
        
        out = torch.cat([rgb, sigma], -1)
        
        # tiks = [i - j for i, j in zip(toks[1:], toks[:-1])]
        # print(tiks)
        return out


    def distill_forward(self, x, sigma_only=False, img_idx=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if img_idx is None:
            img_idx = torch.tensor([0]).to(x.device)

        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if not self.use_scene_code:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        else: 
            img_code = self.scene_code(img_idx).repeat(x.shape[0], 1)
            img_code = self.scene_code_encoding(img_code)

            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir, img_code], -1)
            
        dir_encoding = self.dir_encoding_distill(dir_encoding_input)
        rgb = self.distill(dir_encoding)

        out = torch.cat([rgb, sigma], -1)
        return out

class NeRF_SOS(nn.Module):
    def __init__(self,
                 D=8, 
                 W=256,
                 in_channels_xyz=63, 
                 in_channels_dir=27, 
                 skips=[4], 
                 train_stage=1, 
                 use_scene_code=False, 
                 distribute_color=False
                 ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        train_stage: 
            1 for building grayscale nerf; 
            2 for fix the rgb and sigma mlp, just train the distill mlp
        """
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())
        
        # add exrta rgb distill layer
        self.scene_code_dim = 128
        self.use_scene_code = use_scene_code
        
        self.distribute_color = distribute_color
        if self.use_scene_code:
            # define a learnable latent code
            # self.scene_code = nn.Parameter(torch.randn(1, self.scene_code_dim))
            self.scene_code = None
            self.scene_code_encoding = nn.Sequential(
                                        nn.Linear(self.scene_code_dim, W//2),
                                        nn.ReLU(True), 
                                        nn.Linear(W//2, W//2), 
                                        nn.ReLU(True)
                                        )
            dir_encoding_input_dim = W + in_channels_dir + W//2      
        else:
            dir_encoding_input_dim = W + in_channels_dir


        self.dir_encoding_distill = nn.Sequential(
                                        nn.Linear(dir_encoding_input_dim, W//2),
                                        nn.ReLU(True), 
                                        nn.Linear(W//2, W//2),
                                        nn.ReLU(True)
                                        )

        self.distill = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())
        
        if train_stage == 2:
            print("# freeze all the parameters except self.dir_encoding_distill and self.distill")
            for param in self.parameters():
                param.requires_grad = False
            for param in self.dir_encoding_distill.parameters():
                param.requires_grad = True
            for param in self.distill.parameters():
                param.requires_grad = True
    
    def set_N_imgs(self, N_imgs):
        print("Initializing scene code with N_imgs = ", N_imgs)
        self.scene_code = nn.Embedding(N_imgs, self.scene_code_dim)

    def forward(self, x, sigma_only=False):
        return None
        
    def rgb_forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out


    def distill_forward(self, x, sigma_only=False, img_idx=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if img_idx is None:
            img_idx = torch.tensor([0]).to(x.device)

        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if not self.use_scene_code:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        else: 
            img_code = self.scene_code(img_idx).repeat(x.shape[0], 1)
            img_code = self.scene_code_encoding(img_code)

            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir, img_code], -1)
            
        dir_encoding = self.dir_encoding_distill(dir_encoding_input)
        rgb = self.distill(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out

class NeRF_MVS(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 in_channels_fea=3,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_fea: number of input channels for feature (3 by default)
        skips: add skip connection in the Dth layer
        """
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_fea = in_channels_fea
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        # self.dir_encoding = nn.Sequential(
        #                         nn.Linear(W+in_channels_dir, W//2),
        #                         nn.ReLU(True))
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir+in_channels_fea, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            # input_xyz, input_dir = \
            #     torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
            input_xyz, input_dir, input_feature = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir, self.in_channels_fea], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        # dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding_input = torch.cat([xyz_encoding_final, 
                                        input_dir, 
                                        input_feature], -1)
        
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out