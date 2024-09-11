import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import pandas as pd
from sklearn.model_selection import train_test_split


class CramedDataset(Dataset):

    def __init__(self, args, data, train=True):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.train = train

        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = args.visual_path
        self.audio_feature_path = args.audio_path

        for item in data:
            audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
            visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])

            if os.path.exists(audio_path) and os.path.exists(visual_path):
                self.image.append(visual_path)
                self.audio.append(audio_path)
                self.label.append(class_dict[item[1]])
            else:
                continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

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
        images = torch.zeros((self.args.fps, 3, 224, 224))
        for i in range(self.args.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label

def load_cremad(args, data_root='./Speech_project_Vin/data/'):
    train_csv = os.path.join(data_root, args.dataset + '/train.csv')
    test_csv = os.path.join(data_root, args.dataset + '/test.csv')

    train_df = pd.read_csv(train_csv, header=None)
    train, dev = train_test_split(train_df, test_size=0.1)
    test = pd.read_csv(test_csv, header=None)
    
    train_dataset = CramedDataset(args, train.to_numpy(), True)
    dev_dataset = CramedDataset(args, dev.to_numpy(), False)
    test_dataset = CramedDataset(args, test.to_numpy(), False)

    return train_dataset, dev_dataset, test_dataset
