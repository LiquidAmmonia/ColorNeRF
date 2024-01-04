from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from collections import OrderedDict

from pdb import set_trace as st

# from IPython import embed

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_subset_dict(in_dict,keys):
    if(len(keys)):
        subset = OrderedDict()
        for key in keys:
            subset[key] = in_dict[key]
    else:
        subset = in_dict
    return subset



# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.to(rgb.device)

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.to(rgb.device)

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.to(xyz.device)

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.to(xyz.device)

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).to(lab.device), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.to(lab.device)

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2xyz')
        # embed()

    return out

def rgb2lab(rgb, opt):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-opt.l_cent)/opt.l_norm
    ab_rs = lab[:,1:,:,:]/opt.ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

def lab2rgb(lab_rs, opt):
    l = lab_rs[:,[0],:,:]*opt.l_norm + opt.l_cent
    ab = lab_rs[:,1:,:,:]*opt.ab_norm
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out

def get_colorization_data(data_raw, opt, ab_thresh=5., p=.125, num_points=None):
    data = {}

    data_lab = rgb2lab(data_raw[0], opt)
    data['A'] = data_lab[:,[0,],:,:]
    data['B'] = data_lab[:,1:,:,:]

    if(ab_thresh > 0): # mask out grayscale images
        thresh = 1.*ab_thresh/opt.ab_norm
        mask = torch.sum(torch.abs(torch.max(torch.max(data['B'],dim=3)[0],dim=2)[0]-torch.min(torch.min(data['B'],dim=3)[0],dim=2)[0]),dim=1) >= thresh
        data['A'] = data['A'][mask,:,:,:]
        data['B'] = data['B'][mask,:,:,:]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if(torch.sum(mask)==0):
            return None

    return add_color_patches_rand_gt(data, opt, p=p, num_points=num_points)

def add_color_patches_rand_gt(data,opt,p=.125,num_points=None,use_avg=True,samp='normal'):
# Add random color points sampled from ground truth based on:
#   Number of points
#   - if num_points is 0, then sample from geometric distribution, drawn from probability p
#   - if num_points > 0, then sample that number of points
#   Location of points
#   - if samp is 'normal', draw from N(0.5, 0.25) of image
#   - otherwise, draw from U[0, 1] of image
    N,C,H,W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        pp = 0
        cont_cond = True
        while(cont_cond):
            if(num_points is None): # draw from geometric
                # embed()
                cont_cond = np.random.rand() < (1-p)
            else: # add certain number of points
                cont_cond = pp < num_points
            if(not cont_cond): # skip out of loop if condition not met
                continue

            P = np.random.choice(opt.sample_Ps) # patch size

            # sample location
            if(samp=='normal'): # geometric distribution
                h = int(np.clip(np.random.normal( (H-P+1)/2., (H-P+1)/4.), 0, H-P))
                w = int(np.clip(np.random.normal( (W-P+1)/2., (W-P+1)/4.), 0, W-P))
            else: # uniform distribution
                h = np.random.randint(H-P+1)
                w = np.random.randint(W-P+1)
            
            # add color point
            if(use_avg):
                # embed()
                data['hint_B'][nn,:,h:h+P,w:w+P] = torch.mean(torch.mean(data['B'][nn,:,h:h+P,w:w+P],dim=2,keepdim=True),dim=1,keepdim=True).view(1,C,1,1)
            else:
                data['hint_B'][nn,:,h:h+P,w:w+P] = data['B'][nn,:,h:h+P,w:w+P]

            data['mask_B'][nn,:,h:h+P,w:w+P] = 1

            # increment counter
            pp+=1

    data['mask_B']-=opt.mask_cent

    return data


def get_colorization_data_withmask(data_raw, opt, hint_b, ab_thresh=5.):
    data = {}

    data_lab = rgb2lab(data_raw[0], opt)
    data['A'] = data_lab[:,[0,],:,:]
    data['B'] = data_lab[:,1:,:,:]

    if(ab_thresh > 0): # mask out grayscale images
        thresh = 1.*ab_thresh/opt.ab_norm
        mask = torch.sum(torch.abs(torch.max(torch.max(data['B'],dim=3)[0],dim=2)[0]-torch.min(torch.min(data['B'],dim=3)[0],dim=2)[0]),dim=1) >= thresh
        data['A'] = data['A'][mask,:,:,:]
        data['B'] = data['B'][mask,:,:,:]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if(torch.sum(mask)==0):
            return None

    return add_color_patches_mask(data, opt, hint_b)

def add_color_patches_mask(data,opt,hint_b):
# Add random color points sampled from ground truth based on:
#   Number of points
#   - if num_points is 0, then sample from geometric distribution, drawn from probability p
#   - if num_points > 0, then sample that number of points
#   Location of points
#   - if samp is 'normal', draw from N(0.5, 0.25) of image
#   - otherwise, draw from U[0, 1] of image
    N,C,H,W = data['B'].shape

    # data['hint_B'] = torch.zeros_like(data['B'])
    data['hint_B'] = hint_b
    data['mask_B'] = torch.zeros_like(data['A'])
    data['mask_B'][hint_b[:, [0], :, :] > 0] = 1
    data['mask_B'] -= opt.mask_cent
    # mask_b += opt.mask_cent
    # # find contours
    # mask_b_np = mask_b.squeeze(0).permute(1,2,0).numpy()
    # mask_b_np = np.uint8(mask_b_np)
    # contures, hierarchy = cv2.findContours(mask_b_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # 
    # st()
    # data['hint_B'][mask] = torch.mean(torch.mean(data['B'][mask],dim=2,keepdim=True),dim=1,keepdim=True).view(1,C,1,1)

    return data


def add_color_patch(data,mask,opt,P=1,hw=[128,128],ab=[0,0]):
    # Add a color patch at (h,w) with color (a,b)
    data[:,0,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[0]/opt.ab_norm
    data[:,1,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1.*ab[1]/opt.ab_norm
    mask[:,:,hw[0]:hw[0]+P,hw[1]:hw[1]+P] = 1-opt.mask_cent

    return (data,mask)

def crop_mult(data,mult=16,HWmax=[800,1200]):
    # crop image to a multiple
    # st()
    H,W = data.shape[2:]
    Hnew = int(min(H/mult*mult,HWmax[0]))
    Wnew = int(min(W/mult*mult,HWmax[1]))
    h = (H-Hnew)/2
    w = (W-Wnew)/2

    return data[:,:,h:h+Hnew,w:w+Wnew]

def encode_ab_ind(data_ab, opt):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)

    data_ab_rs = torch.round((data_ab*opt.ab_norm + opt.ab_max)/opt.ab_quant) # normalized bin number
    data_q = data_ab_rs[:,[0],:,:]*opt.A + data_ab_rs[:,[1],:,:]
    return data_q

def decode_ind_ab(data_q, opt):
    # Decode index into ab value
    # INPUTS
    #   data_q      Nx1xHxW \in [0,Q)
    # OUTPUTS
    #   data_ab     Nx2xHxW \in [-1,1]

    data_a = data_q/opt.A
    data_b = data_q - data_a*opt.A
    data_ab = torch.cat((data_a,data_b),dim=1)

    if(data_q.is_cuda):
        type_out = torch.cuda.FloatTensor
    else:
        type_out = torch.FloatTensor
    data_ab = ((data_ab.type(type_out)*opt.ab_quant) - opt.ab_max)/opt.ab_norm

    return data_ab

def decode_max_ab(data_ab_quant, opt):
    # Decode probability distribution by using bin with highest probability
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab         Nx2xHxW \in [-1,1]

    data_q = torch.argmax(data_ab_quant,dim=1)[:,None,:,:]
    return decode_ind_ab(data_q, opt)

def decode_mean(data_ab_quant, opt):
    # Decode probability distribution by taking mean over all bins
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab_inf     Nx2xHxW \in [-1,1]

    (N,Q,H,W) = data_ab_quant.shape
    a_range = torch.range(-opt.ab_max, opt.ab_max, step=opt.ab_quant).to(data_ab_quant.device)[None,:,None,None]
    a_range = a_range.type(data_ab_quant.type())

    # reshape to AB space
    data_ab_quant = data_ab_quant.view((N,int(opt.A),int(opt.A),H,W))
    data_a_total = torch.sum(data_ab_quant,dim=2)
    data_b_total = torch.sum(data_ab_quant,dim=1)

    # matrix multiply
    data_a_inf = torch.sum(data_a_total * a_range,dim=1,keepdim=True)
    data_b_inf = torch.sum(data_b_total * a_range,dim=1,keepdim=True)

    data_ab_inf = torch.cat((data_a_inf,data_b_inf),dim=1)/opt.ab_norm

    return data_ab_inf

def calculate_psnr_np(img1, img2):
    import numpy as np
    SE_map = (1.*img1-img2)**2
    cur_MSE = np.mean(SE_map)
    return 20*np.log10(255./np.sqrt(cur_MSE))

def calculate_psnr_torch(img1, img2):
    SE_map = (1.*img1-img2)**2
    cur_MSE = torch.mean(SE_map)
    return 20*torch.log10(1./torch.sqrt(cur_MSE))
