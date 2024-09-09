import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import yaml
from torch import optim
from model.av_classifier import AVClassifier  # Assuming AVClassifier is defined elsewhere

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def train_epoch(config, epoch, model, device, dataloader, optimizer, scheduler, save_path=None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    for step, (spec, image, label) in (pbar := tqdm(enumerate(dataloader), desc=f'Epoch: {epoch}')):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # Forward pass
        a, v, out = model(spec.unsqueeze(1).float(), image.float())

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        pbar.set_description(f'Epoch: {epoch} Loss: {loss.item():.4f}')

        _loss += loss.item()

    scheduler.step()

    # Save model weights if save_path is specified
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    return _loss / len(dataloader)


def eval(config, model, device, dataloader, test=False):
    softmax = nn.Softmax(dim=1)
    n_classes = config['num_classes']

    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        _loss = 0
        golds = []
        preds = []

        for step, (spec, image, label) in enumerate(dataloader):
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


# Initialize model, optimizer, and scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.nn.DataParallel(AVClassifier(config), device_ids=[0, 1])
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'], momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, config['training']['lr_decay_step'], config['training']['lr_decay_ratio'])

# Training loop
num_epochs = config['training']['epoch']
save_path = config['training']['save_path']

for epoch in range(num_epochs):
    train_loss = train_epoch(config, epoch, model, device, train_dataloader, optimizer, scheduler, save_path=save_path)

    print(f"Train Loss for Epoch {epoch}: {train_loss:.4f}")
    val_loss, wf1 = eval(config, model, device, test_dataloader, test=True)
    print(f"Validation Loss: {val_loss:.4f}, Weighted F1 Score: {wf1:.4f}")
