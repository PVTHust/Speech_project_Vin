# First, install the required libraries (run this in a cell if not already installed)
# !pip install torch torchvision tqdm scikit-learn pyyaml

# Import the necessary libraries
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
from torch import optim
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from model.Ourmodel import AVClassifier  # Import your model
from dataset import CramedDataset  # Assuming you have a dataset class
from model.fusion_method import ConcatFusion, SumFusion, FiLM, GatedFusion  # Import fusion methods

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Training function
def train_epoch(config, epoch, model, device, dataloader, optimizer, scheduler, save_path=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    print("Start training ... ")
    _loss = 0

    for step, (spec, image, label) in tqdm(enumerate(dataloader), desc=f'Epoch: {epoch}', total=len(dataloader)):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        a, v, out = model(spec.unsqueeze(1).float(), image.float())
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
def eval(config, model, device, dataloader, test=False):
    softmax = nn.Softmax(dim=1)
    n_classes = config['num_classes']

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

            a, v, out = model(spec.unsqueeze(1).float(), image.float())
            loss = criterion(out, label)
            _loss += loss.item()

            y_hat = torch.argmax(softmax(out), dim=-1)
            golds.extend(label.cpu().numpy())
            preds.extend(y_hat.cpu().numpy())

        wf1 = f1_score(golds, preds, average='weighted')

        if test:
            print(classification_report(golds, preds))

    return _loss / len(dataloader), wf1

