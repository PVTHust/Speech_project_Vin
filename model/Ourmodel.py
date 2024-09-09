import torch
import torch.nn as nn
import yaml
from model.fusion_method import ConcatFusion, SumFusion, FiLM, GatedFusion
from model.AST import ASTModel
from model.manet import manet

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
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
        self.audio_net = ASTModel(input_tdim=config['input_tdim'], label_dim=64, audioset_pretrain=True)

        # Load visual model (MANet)
        visual_sd = torch.load(config['weight_visual'], map_location='cuda')
        visual_model = manet(num_classes=n_classes)
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


if __name__ == '__main__':
    # Example of how to initialize and run AVClassifier
    av_classifier = AVClassifier(config)

    # Example forward pass with random data
    audio_input = torch.rand(8, 1, 256, 128)  # Random audio input
    visual_input = torch.rand(8, 3, 224, 224)  # Random visual input

    audio_out, visual_out, fusion_out = av_classifier(audio_input, visual_input)
    print(fusion_out.shape)  # Output shape from the fusion
