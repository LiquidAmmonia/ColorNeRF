from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from models.segm.model.vit import VisionTransformer
from models.segm.model.utils import checkpoint_filter_fn
from models.segm.model.decoder import DecoderLinear
from models.segm.model.decoder import MaskTransformer
from models.segm.model.segmenter import Segmenter
import models.segm.utils.torch as ptu


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model




def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")        # vit_tiny_patch16_384

    root_dir = model_cfg.pop('root_dir')
    # breakpoint()

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        print('backbone in timm.visual_t.default_cfgs', backbone)
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(**model_cfg)

    if backbone == 'vit_tiny_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_pth = os.path.join(root_dir, 'resources/vit_tiny_patch16_384.npz')
        model.load_pretrained(pretrained_vit_pth)

    elif backbone == 'vit_small_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_pth = os.path.join(root_dir, 'resources/vit_small_patch16_384.npz')
        model.load_pretrained(pretrained_vit_pth)

    elif backbone == 'vit_base_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_pth = os.path.join(root_dir, 'resources/vit_base_patch16_384.npz')
        model.load_pretrained(pretrained_vit_pth)

    elif backbone == 'vit_large_patch16_384':
        print('backbone:', backbone)
        pretrained_vit_pth = os.path.join(root_dir, 'resources/vit_large_patch16_384.npz')
        model.load_pretrained(pretrained_vit_pth)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        pretrained_vit_pth = os.path.join(root_dir, 'resources/vit_tiny_patch16_224.npz')
        model.load_pretrained(pretrained_vit_pth)

    return model


def create_decoder(encoder, decoder_cfg, backbone):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    if 'vit' in backbone:
        decoder_cfg["d_encoder"] = encoder.d_model      # 192 for vit-tiny
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        if 'vit' in backbone:
            dim = encoder.d_model
            n_heads = dim // 64     # original_dim = 192
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    if 'vit' in model_cfg['backbone']:
        encoder = create_vit(model_cfg)

    decoder = create_decoder(encoder, decoder_cfg, model_cfg['backbone'])
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"], before_classify=decoder_cfg['before_classify'], backbone=model_cfg['backbone'], without_classification=decoder_cfg['without_classification'])

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant
