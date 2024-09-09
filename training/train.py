import torch
import torch.nn as nn
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, classification_report

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None, save_path=None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    for step, (spec, image, label) in (pbar := tqdm(enumerate(dataloader), desc='Epoch: {}: '.format(epoch))):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out = model(spec.unsqueeze(1).float(), image.float())

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        pbar.set_description('Epoch: {} Loss: {:.4f}'.format(epoch, loss.item()))

        _loss += loss.item()

    scheduler.step()

    # Save model weights if save_path is specified
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    return _loss / len(dataloader)

def eval(args, model, device, dataloader, test=False):

    softmax = nn.Softmax(dim=1)

    if args.dataset == 'CREMAD':
        n_classes = 6
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

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

#             a, v, out = model(spec.unsqueeze(1).float(), image.float())
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


from torch import optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.nn.DataParallel( AVClassifier(args), device_ids=[0,1])
# model = AVClassifier(args)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

# Huấn luyện mô hình

num_epochs = args.epoch
save_path = args.save_path

for epoch in range(num_epochs):
    train_loss = train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler, save_path=save_path)

    print(f"Train Loss for Epoch {epoch}: {train_loss:.4f}")
    val_loss = eval(args, model, device, test_dataloader, test=True)