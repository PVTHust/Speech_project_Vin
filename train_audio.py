from model.AST import ASTModel

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchaudio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
import copy
from dataset.CramedDataset import CramedDataset, load_cremad
from model.AST import ASTModel
from tqdm import tqdm

parser = argparse.ArgumentParser(description='train_audio')
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--audio_path', type=str, default='CREMA-D/AudioWAV/', help='Path to audio files')
parser.add_argument('--visual_path', type=str, default='CREMA-D/VideoFlashFPS30/', help='Path to visual files')
parser.add_argument('num_workers', type=int, default=4, help='Number of workers for dataloader')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--save_path', type=str, default='/content/kaggle-working/', help='Path to save the model')
args = parser.parse_args()

device = torch.device(args.device)
train_dataset, dev_dataset, test_dataset = load_cremad(args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)    
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# define model
audio_model = ASTModel(input_tdim=256,label_dim=64, audioset_pretrain=True)
audio_model = torch.nn.DataParallel(audio_model, device_ids=[0, 1])
audio_model.to(device)

def train_epoch_audio(epoch, model, device, dataloader, optimizer, scheduler, save_path=args.save_path):
    criterion = nn.CrossEntropyLoss()
    model.train()
    print("Start training ... ")
    _loss = 0

    for step, (spec, image, label) in tqdm(enumerate(dataloader), desc=f'Epoch: {epoch}', total=len(dataloader)):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        out = model(spec.unsqueeze(1).float(), image.float())
        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        _loss += loss.item()

    scheduler.step()

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    return _loss / len(dataloader)

# Evaluation function
def eval(args, model, device, dataloader, test=False):
    softmax = nn.Softmax(dim=1)
    n_classes = args.num_classes

    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        _loss = 0
        golds = []
        preds = []

        for step, (spec, image, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out = model(spec.unsqueeze(1).float(), image.float())
            loss = criterion(out, label)
            _loss += loss.item()

            y_hat = torch.argmax(softmax(out), dim=-1)
            golds.extend(label.cpu().numpy())
            preds.extend(y_hat.cpu().numpy())

        wf1 = f1_score(golds, preds, average='weighted')

        if test:
            print(classification_report(golds, preds))

    return _loss / len(dataloader), wf1



optimizer = torch.optim.Adam(audio_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


for epoch in range(args.epochs):
    train_loss = train_epoch_audio(epoch, audio_model, device, train_dataloader, optimizer, scheduler, save_path=args.save_path)
    print(f"Epoch {epoch} - Train Loss: {train_loss}")

    val_loss, wf1 = eval(args, audio_model, device, dev_dataloader)
    print(f"Epoch {epoch} - Dev Loss: {val_loss}, Dev F1: {wf1}")

    if args.save_path:
        torch.save(audio_model.state_dict(), os.path.join(args.save_path, f"audio_model_epoch_{epoch}.pth"))
        print(f"Model weights saved to {args.save_path}")
