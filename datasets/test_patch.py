import torch
import numpy as np

from pdb import set_trace as st

if __name__ == '__main__':
    # root_dir = '/userhome/chengyean/ARF-svox2/data/llff/flower'
    # dataset = HfaiLLFFRefDataset(root_dir=root_dir)
    # print(len(dataset))
    
    # test selected inds
    
    
    def _get_select_inds(N_samples_sqrt, 
                         method='central'):
        
        """
        method: 
            'central':
            'random': 
        """
        orig_w, orig_h = torch.meshgrid([torch.linspace(-1, 1, N_samples_sqrt),
                               torch.linspace(-1, 1, N_samples_sqrt)])
        h = orig_h.unsqueeze(2)
        w = orig_w.unsqueeze(2)
        
        if method == 'random':
            min_scale = 0.1
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
        
        
        elif method == 'central':
            min_scale = 0.5
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
        
    from matplotlib import pyplot as plt
    import cv2
    
    for method in ['central', 'random']:
        
        white = np.zeros((256, 256, 3))
        plt.figure()
        for i in range(20):
            inds = _get_select_inds(64, method=method)
            # draw rect in white
            up_left = inds[0, 0] * 128 + 128
            down_right = inds[-1, -1] * 128 + 128
            
            # to int
            up_left = torch.clip(up_left.int(), 0, 255)
            down_right = torch.clip(down_right.int(), 0, 255)
            
            cv2.rectangle(white, 
                        (up_left[0].item(), up_left[1].item()),
                        (down_right[0].item(), down_right[1].item()), 
                        (0, 1, 0), 1)
            
        plt.imsave(f'{method}_verbose.png', white)
        plt.clf()
    