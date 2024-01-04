from torch import nn
import torchvision
import torch

from pdb import set_trace as st
import math 

from kornia.color import rgb_to_lab
import torch.nn.functional as F

from kornia.losses import total_variation

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels, add_mask=None):
        B, n_cls, H, W = outputs.shape
        reshape_out = outputs.permute(0, 2, 3, 1).contiguous().view(B*H*W, n_cls)
        reshape_label = labels.permute(0, 2, 3, 1).contiguous().view(B*H*W, n_cls)       # [-1, 313]
        after_softmax = F.softmax(reshape_out, dim=1)
        # mask = add_mask.view(-1, n_cls)
        mask = after_softmax.clone()
        after_softmax = after_softmax.masked_fill(mask < 1e-5, 1)
        out_softmax = torch.log(after_softmax)

        norm = reshape_label.clone()
        norm = norm.masked_fill(reshape_label < 1e-5, 1)
        log_norm = torch.log(norm)

        loss = -torch.sum((out_softmax - log_norm) * reshape_label) / (B*H*W)
        return loss




class EdgeEnhanceLoss(nn.Module):
    def __init__(self, coef=1):
        """
        1. extract ab channel from image
        2. apply sobel filter to ab channel
        """
        super().__init__()
        self.kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.mse = nn.MSELoss(reduction='mean')
        self.coef = coef

    def _get_sobel(self, img):
        # img: [1, 1, H, W]
        # apply sobel filter
        self.kernel = self.kernel.to(img.device)
        sobelx = F.conv2d(img, self.kernel, padding=1)
        sobely = F.conv2d(img, self.kernel.transpose(2, 3), padding=1)
        # return sqrt(sobelx^2 + sobely^2)
        sobel = torch.sqrt(sobelx ** 2 + sobely ** 2)
        # normalize to [0, 1]
        sobel = (sobel - torch.min(sobel)) / (torch.max(sobel) - torch.min(sobel))
        return sobel

    def _get_ab(self, img):
        # img: [1, 3, H, W]
        # convert to lab and return img_a, img_b
        lab = rgb_to_lab(img)
        # extract ab channel
        img_a = lab[:, 1:2, :, :]
        img_b = lab[:, 2:3, :, :]
        return img_a, img_b
    
    def forward(self, img, gt):
        # img: [1, 3, H, W]
        # gt: [1, 3, H, W]
        # get ab channel
        
        img_a, img_b = self._get_ab(img)
        gt_a, gt_b = self._get_ab(gt)

        loss = 0
        # get sobel for a channel
        sobel_img_a = self._get_sobel(img_a)
        sobel_gt_a = self._get_sobel(gt_a)

        loss += self.mse(sobel_img_a, sobel_gt_a)
        # get sobel for a channel
        sobel_img_b = self._get_sobel(img_b)
        sobel_gt_b = self._get_sobel(gt_b)
        loss += self.mse(sobel_img_b, sobel_gt_b)
        return loss

    def full_forward(self, inputs, targets, prefix='rgb'):
        # img: [1, 3, H, W]
        # gt: [1, 3, H, W]
        img = inputs[f'{prefix}_coarse']
        loss = self._get_edge_loss(img, targets)
        if f'{prefix}_fine' in inputs:
            img = inputs[f'{prefix}_fine']
            loss += self._get_edge_loss(img, targets)
        return self.coef * loss

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.SmoothL1Loss(reduction='mean')
        # self.loss = nn.MSELoss(reduction='mean')
    
    def _nomalize_ab(self, ab):
        # normalize ab channel to [-1, 1]
        _ab = ab / 128
        _ab = torch.clamp(_ab, -1, 1)
        return _ab

    def forward(self, inputs, targets, prefix='rgb', suffix='', normalize=False):
        loss = self.loss(inputs[f'{prefix}_coarse{suffix}'], targets)
        if f'{prefix}_fine' in inputs:
            if not normalize:
                loss += self.loss(inputs[f'{prefix}_fine{suffix}'], targets)
            else:
                loss += self.loss(self._nomalize_ab(inputs[f'{prefix}_fine{suffix}']),
                                  self._nomalize_ab(targets))
        return self.coef * loss

class TotalVariationLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x):
        # x: [1, 3, H, W]
        struc_loss = total_variation(x, self.reduction)
        return torch.mean(struc_loss)
    

class ABLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')
        
        
    def forward(self, input, target):
        # assume input and target are in lab space
        # TODO: check if need to change to ab range to [-1, 1], now it is [-128, 127]
        assert input.shape[1] == 2
        assert target.shape[1] == 2
        return self.coef * self.loss(input, target)

class GrayLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')
        self.coarse_rgb_w = 0.5

    def _to_gray(self, rgb):
        return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    
    def forward(self, inputs, targets, coarse_rgb=None):
        loss = self.loss(self._to_gray(inputs['rgb_coarse']), 
                         self._to_gray(targets))
        
        if coarse_rgb is not None:
            loss += self.coarse_rgb_w * self.loss(inputs['rgb_coarse'], 
                                                  coarse_rgb)
            
        if 'rgb_fine' in inputs:
            loss += self.loss(self._to_gray(inputs['rgb_fine']), 
                              self._to_gray(targets))
            if coarse_rgb is not None:
                loss += self.coarse_rgb_w * self.loss(inputs['rgb_fine'], 
                                                      coarse_rgb)

        return self.coef * loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.L1Loss(reduction='mean')
        self.vgg = torchvision.models.vgg16(pretrained=True).eval()
        
        for p in self.vgg.parameters():
            p.requires_grad = False
        
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        
        
        self.vgg_device_init = False
        self.chunk = 4096
        self.w_vgg = 1e-3
    
    def _to_gray(self, rgb):
        return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        
    def _get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)
            if ix == final_ix:
                break
        return outputs
    
    def _vgg_forward(self, gt, pred, blocks=[2, ]):
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], 
                         [18, 20, 22], [25, 27, 29]]
        blocks.sort()

        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self._get_feats(pred, all_layers)
        with torch.no_grad():
            gt_feats_all = self._get_feats(gt, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        loss = 0.
        for block in blocks:
            layers = block_indexes[block]

            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            content_feats = torch.cat(
                [gt_feats_all[ix_map[ix]] for ix in layers], 1)
            loss += torch.mean((content_feats - x_feats) ** 2)

        return loss

    def _view_image(self, img, HW=None):
        if HW is None:
            H, W = self.H, self.W
        else:
            H, W = HW, HW
        return img.view(-1, 3, H, W)

    def forward(self, inputs, targets, ref_ls=None):
        if ref_ls is None:
            raise ValueError('VGG perceptual loss requires a reference image')
        
        if self.vgg_device_init == False:
            self.vgg.to(inputs['rgb_coarse'].device)
            self.vgg_device_init = True
            
        # Gray scale loss with original image  
        loss = self.loss(self._to_gray(inputs['rgb_coarse']), 
                         self._to_gray(targets))
        if 'rgb_fine' in inputs:
            loss += self.loss(self._to_gray(inputs['rgb_fine']), 
                              self._to_gray(targets))
        
        # perceptual loss with reference image
        ref = ref_ls[0]
        self.H, self.W = ref.shape[0], ref.shape[1]
        ref_gpu = ref.unsqueeze(0).reshape(1, 3, self.H, self.W).to(
            inputs['rgb_coarse'].device)
        
        chunk_size = None
        if inputs['rgb_coarse'].shape[0] == self.chunk:
            chunk_size = int(math.sqrt(self.chunk))
            # random crop ref
            ref_gpu = torchvision.transforms.Resize(
                (chunk_size, chunk_size))(ref_gpu)
            
            
        if not (inputs['rgb_coarse'].shape[0] < self.chunk and 
                inputs['rgb_coarse'].shape[0] < self.H * self.W):
            loss += self.w_vgg * self._vgg_forward(
                self._view_image(inputs['rgb_coarse'], HW=chunk_size), ref_gpu)
            if 'rgb_fine' in inputs:
                loss += self.w_vgg * self._vgg_forward(
                    self._view_image(inputs['rgb_fine'], HW=chunk_size), ref_gpu)
            
        return self.coef * loss


loss_dict = {'color': ColorLoss, 
             'gray': GrayLoss, 
             'gray_vgg': VGGPerceptualLoss, 
             'edge': EdgeEnhanceLoss, 
             'ab_loss': ABLoss,
             'ce2d': CrossEntropyLoss2d,
             'tv_loss': TotalVariationLoss,
             }


if __name__ == '__main__':
    tv = torch.rand(5, 3, 256, 256)
    loss = TotalVariationLoss()
    out = loss(tv)