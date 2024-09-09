import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import yaml

# Load the configuration from the YAML file
with open('/kaggle/working/Speech_project_Vin/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
