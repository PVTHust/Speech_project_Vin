#@title Trainning
import torch 
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

def train_epoch(epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    for step, (spec, image, label) in (pbar := tqdm(enumerate(dataloader), desc='Epoch: {}: '.format(epoch))):
        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out = model(spec.float(), image.float())

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        pbar.set_description('Epoch: {} Loss: {:.4f}'.format(epoch, loss.item()))

        _loss += loss.item()

    scheduler.step()

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

            a, v, out = model(spec.float(), image.float())


            loss = criterion(out, label)
            _loss += loss.item()

            y_hat = torch.argmax(softmax(out), dim=-1)
            golds.extend(label.cpu().numpy())
            preds.extend(y_hat.cpu().numpy())

        wf1 = f1_score(golds, preds, average='weighted')

        if test:
            print(classification_report(golds, preds))
    return _loss / len(dataloader), wf1