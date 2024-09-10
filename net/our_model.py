import torch
import torch.nn as nn
import os
import sys
from net.fusion_method import ConcatFusion, SumFusion, FiLM, GatedFusion
from net.AST import ASTModel
from net.manet import manet, RecorderMeter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import get_args

args = get_args()

class AVClassifier(nn.Module):
    def __init__(self):
        super(AVClassifier, self).__init__()

        fusion_method = args.fusion_type
        n_classes = args.num_classes

        # Select fusion module based on args
        if fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=args.fusion_input_dim, output_dim=n_classes)
        elif fusion_method == 'concat':
            self.fusion_module = ConcatFusion(input_dim=args.fusion_input_dim, output_dim=n_classes)
        elif fusion_method == 'film':
            self.fusion_module = FiLM(input_dim=args.fusion_input_dim, output_dim=n_classes)
        elif fusion_method == 'gated':
            self.fusion_module = GatedFusion(input_dim=args.fusion_input_dim, output_dim=n_classes)
        else:
            raise ValueError("Invalid fusion method in args")

        # Initialize audio model (AST)
        self.audio_net = ASTModel(input_tdim=256, label_dim=64, audioset_pretrain=True)

        # Load visual model (MANet)
        visual_sd = torch.load(args.weight_visual, map_location='cuda')
        visual_model = manet()
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
