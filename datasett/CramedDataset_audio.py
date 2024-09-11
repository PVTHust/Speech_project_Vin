import copy
import csv
import os
import pickle
import torchaudio
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import pandas as pd
from sklearn.model_selection import train_test_split


class CramedDataset_audio(Dataset):

    def __init__(self, args, data, train=True):
        self.args = args
        self.audio = []
        self.label = []
        self.train = train
        self.audio_length = 256

        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        # self.visual_feature_path = args.visual_path
        self.audio_feature_path = args.audio_path

        for item in data:
            audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
            # visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])

            if os.path.exists(audio_path):
                # self.image.append(visual_path)
                self.audio.append(audio_path)
                self.label.append(class_dict[item[1]])
            else:
                continue


    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):

        # audio
        waveform, sr = torchaudio.load(self.audio[idx])
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
        )

        n_frames = fbank.shape[0]
        p = self.audio_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.audio_length, :]

        fbank = fbank.unsqueeze(0)

        # label
        label = self.label[idx]

        return fbank, label

def load_cremad(args, data_root='./Speech_project_Vin/data/'):
    train_csv = os.path.join(data_root, args.dataset + '/train.csv')
    test_csv = os.path.join(data_root, args.dataset + '/test.csv')

    train_df = pd.read_csv(train_csv, header=None)
    train, dev = train_test_split(train_df, test_size=0.1)
    test = pd.read_csv(test_csv, header=None)
    
    train_dataset = CramedDataset_audio(args, train.to_numpy(), True)
    dev_dataset = CramedDataset_audio(args, dev.to_numpy(), False)
    test_dataset = CramedDataset_audio(args, test.to_numpy(), False)

    return train_dataset, dev_dataset, test_dataset
