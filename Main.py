import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vat import VATLoss
from utils import accuracy, Net
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, Sampler

epochs = 100 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xi = .1
eps = 1.
ip = 1
alpha=1e-3
lr=1e-3

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=10)

validset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=400,
                                         shuffle=False, num_workers=10)

model_path = 'vat.pt'
model = Net().to(device)
optimizer =optim.Adam(model.parameters(), lr=lr)

#Baseline encoder + decoder
best_top = 0.

best_epoch = 0
nanFlag = 0


for epoch in range(1, epochs + 1):
    acc = 0
    correct = 0
    total = 0
    valid_loss = 0
    valid_acc= 0
    valid_acc_minority =0

    improved_str = " "
    
    for batched_x, batched_label in train_loader:
        model.train()
        batched_x = batched_x.to(device)   
        vat_loss = VATLoss(xi=xi, eps=eps, ip=ip)
        cross_entropy = nn.CrossEntropyLoss()
        lds = vat_loss(model, batched_x)
        output = model(batched_x)
        classification_loss = cross_entropy(output, batched_label.to(device))
        loss = classification_loss + alpha * lds
        loss.backward()
        optimizer.step()

        
    for i, (val_s, val_label) in enumerate(valid_loader):
        val_s =  val_s.to(device)
        val_pred = model(val_s)
        loss = nn.CrossEntropyLoss()
        batched_ce = loss(val_pred, val_label.to('cuda')) 
        batched_correct = (np.argmax(val_pred.detach().cpu().numpy(), 1)==\
                           val_label.detach().cpu().numpy()).sum()
        batched_total = len(val_label)
        valid_loss += batched_ce
        total += batched_total
        correct +=batched_correct
        acc += accuracy(val_pred.detach().cpu(), val_label.detach(), topk=(5,))[0]
        
    if best_top <= correct/total:
        best_epoch = epoch
        best_top = correct/total
        torch.save(model.state_dict(), model_path)
        improved_str = "*"        
          
    print('====> Valid CE loss: {:.4f}\t \t Acc: {:.4f} \t Improved: {}'.format(
        valid_loss/(i+1),  correct/total, improved_str))
    print('=====> Top 5 Acc', acc.item()/(i+1))
            

    if epoch - best_epoch >=20:
        print('Model stopped due to early stopping')
        break