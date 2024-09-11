import argparse
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from dataset.CramedDataset import load_cremad
from model.AST import ASTModel
import sys


def train_epoch(model, dataloader, optimizer, scheduler, save_path=None):
    """
    Train the model for one epoch on the provided data.
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    _loss = 0

    for step, (spec, image, label) in tqdm(enumerate(dataloader), desc='Training', total=len(dataloader)):
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


def eval(model, dataloader, test=False):
    """
    Evaluate the model on validation or test set.
    """
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    _loss = 0
    golds = []
    preds = []

    with torch.no_grad():
        model.eval()

        for _, (spec, image, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
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





optimizer = torch.optim.Adam(audio_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(args.epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, save_path=args.save_path)
    print(f"Epoch {epoch} - Train Loss: {train_loss}")

    val_loss, wf1 = eval(audio_model, dev_dataloader)
    print(f"Epoch {epoch} - Dev Loss: {val_loss}, Dev F1: {wf1}")

    if args.save_path:
        torch.save(audio_model.state_dict(), os.path.join(args.save_path, f"audio_model_epoch_{epoch}.pth"))
        print(f"Model weights saved to {args.save_path}")
