
import os
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from sklearn.model_selection import train_test_split
import torchaudio
import sys





class CramedDataset(Dataset):

    def __init__(self, args,  data, train=True):
        self.image = []
        self.audio = []
        self.label = []
        self.train = train
        self.audio_length = 256


        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.visual_feature_path = args.visual_path
        self.audio_feature_path = args.audio_path

        for item in data:
            audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
            visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(args.fps, item[0]))

            if os.path.exists(audio_path) and os.path.exists(visual_path):
                self.image.append(visual_path)
                self.audio.append(audio_path)
                self.label.append(class_dict[item[1]])
            else:
                continue

        self.args = args    

    def __len__(self):
        return len(self.image)

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

        if self.train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()

        img = Image.open(os.path.join(self.image[idx], image_samples[select_index[0]])).convert('RGB')
        img = transform(img)

        # label
        label = self.label[idx]

        return fbank, img, label

def load_cremad(args):
    train_csv = args.train_csv
    test_csv = args.test_csv
    train_df = pd.read_csv(train_csv, header=None)
    train, dev = train_test_split(train_df, test_size=0.1)
    test = pd.read_csv(test_csv, header=None)

    train_dataset = CramedDataset(args, train.to_numpy(), True)
    dev_dataset = CramedDataset(args, dev.to_numpy(), False)
    test_dataset = CramedDataset(args, test.to_numpy(), False)

    return train_dataset, dev_dataset, test_dataset

