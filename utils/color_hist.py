import cv2
import os

import numpy as np
from pdb import set_trace as st

class ColorHist:
    def __init__(self, 
                 thres=0.92, 
                 regitser_num_max=5, 
                 bins=8, 
                 force_accept=True):
        self.bins = bins
        self.thres = thres
        self.src_img_ls = []
        self.src_img_tensor_ls = []
        self.meter = [0, 0]
        self.reject_thres = 0.75
        self.reject_iter = 0
        self.regitser_num_max = regitser_num_max
        self.force_accept = force_accept
    
    def _get_hist(self, img):
        # img: [H, W, 3]
        # return hist of img
        hist = cv2.calcHist([img], [0, 1, 2], None, [self.bins] * 3, [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def manual_register(self, img_path, exp_name):
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(self.src_img_ls) < self.regitser_num_max:
            self.src_img_ls.append(img)
        # self.src_img = img
        # save_dir = os.path.join('logs', exp_name)
        # os.makedirs(save_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(save_dir, f'src_img_{len(self.src_img_ls)}.png'), img)
        print(f'MANUALLY Registered source image, saved in {len(self.src_img_ls)}')
        return len(self.src_img_ls), self.regitser_num_max
    
    def reigister_source_img(self, img_tensor, exp_name):
        img = img_tensor.cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(self.src_img_ls) < self.regitser_num_max:
            self.src_img_ls.append(img)
            img_tensor_ = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).cpu()
            self.src_img_tensor_ls.append(img_tensor_)
        # self.src_img = img
        # save_dir = os.path.join('logs', exp_name)
        # os.makedirs(save_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(save_dir, f'src_img_{len(self.src_img_ls)}.png'), img)
        print(f'Registered source image, saved index {len(self.src_img_ls)}')
        return len(self.src_img_ls), self.regitser_num_max
    
    def compare(self, img1, img2):
        # img1: [H, W, 3]
        # img2: [H, W, 3]
        # return hist similarity
        hist1 = self._get_hist(img1)
        hist2 = self._get_hist(img2)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def color_similarity(self, img1_tensor, verbose=False):
        # img1_tensor: [1, 3, H, W]
        # img2_tensor: [1, 3, H, W]
        img1 = img1_tensor.cpu().numpy()
        # img2 = img2_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img1 = (img1 * 255).astype(np.uint8)
        # img2 = (img2 * 255).astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        res_ls = [self.compare(img1, img0) for img0 in self.src_img_ls]
        res = max(res_ls)

        self.meter[0] += 1
        if res > self.thres:
            self.meter[1] += 1
        reject_ratio = self.get_match_ratio()
        if reject_ratio < self.reject_thres:
            if self.force_accept:
                if verbose:
                    print(f'Reject ratio too low: {reject_ratio}, forcing accept')
                self.meter[1] += 1
                return True, res
        return res > self.thres, res
    
    def get_match_ratio(self):
        if self.meter[0] == 0:
            return 0
        return self.meter[1] / self.meter[0]


if __name__ == '__main__':
    root_dir = '/userhome/chengyean/ARF-svox2/data/llff/flower_CT2/images_8'

    img1_dir = os.path.join(root_dir, 'IMG_2962.JPG')
    img1 = cv2.imread(img1_dir)
    
    out_dir = '/userhome/chengyean/ct2_sos/utils/teaser_colorhist/flower_CT2'
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, 'raw.png'), img1)

    # save the least 5 similar images
    last_5 = []
    hist = ColorHist()
    for img_name in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, img_name)
        img2 = cv2.imread(img_dir)
        sim = hist.compare(img1, img2)
        print(img_name, hist.compare(img1, img2))
        if len(last_5) < 5:
            last_5.append((img_name, sim))
            last_5.sort(key=lambda x: -x[1])
        else:
            if sim > last_5[0][1]:
                last_5[0] = (img_name, sim)
                last_5.sort(key=lambda x: -x[1])     

    for img_name, sim in last_5:
        img_dir = os.path.join(root_dir, img_name)
        img2 = cv2.imread(img_dir)
        cv2.imwrite(os.path.join(out_dir, str(sim) + img_name), img2)
        