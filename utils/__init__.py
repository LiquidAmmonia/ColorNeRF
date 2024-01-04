import torch
# optimizer
from torch.optim import SGD, Adam
import torch_optimizer as optim
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *

from pdb import set_trace as st

from kornia.color import rgb_to_lab, lab_to_rgb

def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else: # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = get_parameters(models)
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr, 
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps, 
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = optim.RAdam(parameters, lr=hparams.lr, eps=eps, 
                                weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = optim.Ranger(parameters, lr=hparams.lr, eps=eps, 
                                 weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer, 
                             lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, 
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', words_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    # st()
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for word in words_to_ignore:
            # if k.startswith(prefix):
            if word in k:
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
        
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', words_to_ignore=[]):
    if not ckpt_path:
        return
    model_dict = model.state_dict()
    
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, words_to_ignore)
    # st()
    if 'distill.0.weight' in checkpoint_.keys() and 'distill.weight' in model_dict.keys():
        # bug fix for wrong key name
        checkpoint_['distill.weight'] = checkpoint_.pop('distill.0.weight')
        checkpoint_['distill.bias'] = checkpoint_.pop('distill.0.bias')
            
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

def _modify_shape(img):
    """MOdifys the shape of the image to be [B, 3, H, W]

    Args:
        img (tensor): 

    """
    shape = img.shape
    shape_code = 0
    if not len(shape) == 4:
        if len(shape) == 2:
            # img = [B, 3]
            assert img.shape[1] == 3
            img_ext = img[:, :, None, None]
            shape_code = 1
        elif len(shape) == 3:
            # img = [H, W, 3]
            assert img.shape[2] == 3
            img_ext = img.permute(2, 0, 1)[None, :, :, :]
            shape_code = 2
        else:
            raise ValueError(f'img shape not recognized! {shape}')
    else:
        img_ext = img
    return img_ext, shape_code

def rgb2lab(img):
    img_ext, shape_code = _modify_shape(img)
    lab = rgb_to_lab(img_ext)
    if shape_code == 1:
        lab = lab[:, :, 0, 0]
    elif shape_code == 2:
        lab = lab[0].permute(1, 2, 0)        
    return lab

def lab2rgb(img):
    img_ext, shape_code = _modify_shape(img)
    rgb = lab_to_rgb(img_ext)
    if shape_code == 1:
        rgb = rgb[:, :, 0, 0]
    elif shape_code == 2:
        rgb = rgb[0].permute(1, 2, 0)        
    return rgb

