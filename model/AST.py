import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
from timm.models.layers import trunc_normal_
import yaml
from model.block_audio import PatchEmbed

class ASTModel(nn.Module):
    def __init__(self, config):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if config['ASTModel']['verbose']:
            print('---------------AST Model Summary---------------')
            print(f"ImageNet pretraining: {config['ASTModel']['imagenet_pretrain']}, AudioSet pretraining: {config['ASTModel']['audioset_pretrain']}")

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if not config['ASTModel']['audioset_pretrain']:
            model_size = config['ASTModel']['model_size']
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=config['ASTModel']['imagenet_pretrain'])
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=config['ASTModel']['imagenet_pretrain'])
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=config['ASTModel']['imagenet_pretrain'])
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=config['ASTModel']['imagenet_pretrain'])
            else:
                raise ValueError('Model size must be one of tiny224, small224, base224, base384.')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.original_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, config['ASTModel']['label_dim']))

            f_dim, t_dim = self.get_shape(config['ASTModel']['fstride'], config['ASTModel']['tstride'], config['ASTModel']['input_fdim'], config['ASTModel']['input_tdim'])
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            if config['ASTModel']['verbose']:
                print(f"Frequency stride={config['ASTModel']['fstride']}, Time stride={config['ASTModel']['tstride']}")
                print(f"Number of patches={num_patches}")

            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(config['ASTModel']['fstride'], config['ASTModel']['tstride']))
            if config['ASTModel']['imagenet_pretrain']:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if config['ASTModel']['imagenet_pretrain']:
                new_pos_embed = self.resize_positional_embedding(self.v.pos_embed[:, 2:, :].detach(), f_dim, t_dim)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sdi = torch.load(config['weight_audio'], map_location=device)
            audio_model = ASTModel(config)
            # audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sdi, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, config['ASTModel']['label_dim']))

            f_dim, t_dim = self.get_shape(config['ASTModel']['fstride'], config['ASTModel']['tstride'], config['ASTModel']['input_fdim'], config['ASTModel']['input_tdim'])
            num_patches = f_dim * t_dim

            if config['ASTModel']['verbose']:
                print(f"Frequency stride={config['ASTModel']['fstride']}, Time stride={config['ASTModel']['tstride']}")
                print(f"Number of patches={num_patches}")

            new_pos_embed = self.resize_positional_embedding(self.v.pos_embed[:, 2:, :].detach(), f_dim, t_dim)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def resize_positional_embedding(self, new_pos_embed, f_dim, t_dim):
        if t_dim <= self.original_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.original_hw / 2) - int(t_dim / 2): int(self.original_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.original_hw, t_dim), mode='bilinear')
        if f_dim <= self.original_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.original_hw / 2) - int(f_dim / 2): int(self.original_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        return new_pos_embed.reshape(1, self.original_embedding_dim, f_dim * t_dim).transpose(1, 2)

    @autocast()
    def forward(self, x):
        x = x.transpose(2, 3)
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
        x = (x[:, 0] + x[:, 1]) / 2
        return self.mlp_head(x)

if __name__ == '__main__':
    # Load config from YAML file
    with open('/kaggle/working/Speech_project_Vin/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ast_mdl = ASTModel(config)
    test_input = torch.rand([16, 1, config['ASTModel']['input_tdim'], config['ASTModel']['input_fdim']])
    test_output = ast_mdl(test_input)
    print(test_output.shape)
