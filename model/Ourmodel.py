import torch
import torch.nn as nn
import yaml
from model.fusion_method import ConcatFusion, SumFusion, FiLM, GatedFusion
from model.AST import ASTModel
from model.manet import manet, RecorderMeter

# Load configuration from YAML file
with open('/kaggle/working/Speech_project_Vin/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class AVClassifier(nn.Module):
    def __init__(self, config):
        super(AVClassifier, self).__init__()
        fusion_method = config['fusion']['type']
        n_classes = config['num_classes']

        # Select fusion module based on config
        if fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=config['fusion']['input_dim'], output_dim=n_classes)
        elif fusion_method == 'concat':
            self.fusion_module = ConcatFusion(input_dim=config['fusion']['input_dim'], output_dim=n_classes)
        elif fusion_method == 'film':
            self.fusion_module = FiLM(input_dim=config['fusion']['input_dim'], output_dim=n_classes)
        elif fusion_method == 'gated':
            self.fusion_module = GatedFusion(input_dim=config['fusion']['input_dim'], output_dim=n_classes)
        else:
            raise ValueError("Invalid fusion method in config")

        # Initialize audio model (AST)
        self.audio_net = ASTModel( input_tdim = 256,label_dim=64, audioset_pretrain=True)

        # Load visual model (MANet)
        visual_sd = torch.load(config['weight_visual'], map_location='cuda')
        visual_model = torch.nn.DataParallel(manet(), device_ids=[0, 1])
        visual_model.load_state_dict(visual_sd, strict=False)
        self.visual_net = visual_model

    def forward(self, audio, visual):
        # Process audio input
        audio = audio.squeeze(2).expand(-1, 1, -1, -1)
        a = self.audio_net(audio)

        # Process visual input
        v = self.visual_net(visual, return_embedding=True)

        # Fusion of audio and visual
        a, v, out = self.fusion_module(a, v)

        return a, v, out
