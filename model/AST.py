import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.models.layers import to_2tuple, trunc_normal_
from model.block_audio import PatchEmbed
import yaml

# Load the configuration from the YAML file
with open('/kaggle/working/Speech_project_Vin/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    """
    def __init__(self, config, label_dim=527, verbose=True):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(config['ASTModel']['imagenet_pretrain']), str(config['ASTModel']['audioset_pretrain'])))

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        fstride = config['ASTModel']['fstride']
        tstride = config['ASTModel']['tstride']
        input_fdim = config['ASTModel']['input_fdim']
        input_tdim = config['ASTModel']['input_tdim']
        imagenet_pretrain = config['ASTModel']['imagenet_pretrain']
        audioset_pretrain = config['ASTModel']['audioset_pretrain']
        model_size = config['ASTModel']['model_size']

        if not audioset_pretrain:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim), 
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            # Get intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            if verbose:
                print(f'Frequency stride={fstride}, time stride={tstride}')
                print(f'Number of patches={num_patches}')

            # Adjust the projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # Adjust the positional embedding
            if imagenet_pretrain:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)

                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')

                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast('cuda')
    def forward(self, x):
        x = x.transpose(2, 3)  # Expected input shape: (batch_size, time_frame_num, frequency_bins)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)

        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2  # Combine cls and dist tokens
        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    # Load the configuration
    input_tdim = config['input_tdim']
    ast_mdl = ASTModel(config, label_dim=64, audioset_pretrain=config['audioset_pretrain'])

    # Test input with random data
    test_input = torch.rand([16, 1, input_tdim, 128])
    test_output = ast_mdl(test_input)

    print(test_output.shape)  # Should be in shape [batch_size, label_dim], i.e., [16, 64]
