import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from model.fusion_method import ConcatFusion
from model.AST import ASTModel
from model.manet import manet
class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()
        fusion = args.fusion_method
        n_classes = 7
        self.fusion_module = ConcatFusion(input_dim=1088, output_dim=n_classes)
        self.audio_net = ASTModel(input_tdim=args.input_tdim, label_dim=64, audioset_pretrain=True)
        sd = torch.load(args.weight_visual, map_location='cuda')
        visual_model = manet(num_classes=n_classes)
        visual_model.load_state_dict(sd, strict=False)
        self.visual_net = visual_model
    def forward(self, audio, visual):
        audio = audio.squeeze(2).expand(-1, 1, -1, -1)
        v = self.visual_net(visual, return_embedding=True)
        
        a = self.audio_net(audio)

        a, v, out = self.fusion_module(a, v)
        return a, v, out
