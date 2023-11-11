import os

import kornia.augmentation as K
import numpy as np
import pandas as pd
import torch
from skimage import exposure
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset, Subset

import config

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

@torch.no_grad()
def evaluate(model, device, val_loader, criterion):
    # adapted from CS-433 Machine Learning Exercises
    model.eval()
    test_loss = 0
    correct = 0
    target_tot = []
    pred_tot = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # Assuming this is a pixel-wise classification

        correct += pred.eq(target.view_as(pred)).sum().item()
        target_tot.extend(target.view(-1).cpu().numpy())
        pred_tot.extend(pred.view(-1).cpu().numpy())
        test_loss += criterion(output, target).item() * len(data)

    test_loss /= len(val_loader.dataset)

    # Calculate Metrics
    i = np.logical_and(target_tot, pred_tot)
    u = np.logical_or(target_tot, pred_tot)
    iou = i.sum()/u.sum()
    dice = 2.0 * i.sum() / (target_tot.sum() + pred_tot.sum())
    precision = precision_score(target_tot, pred_tot, average='macro')
    recall = recall_score(target_tot, pred_tot, average='macro')
    f1 = f1_score(target_tot, pred_tot, average='macro')
    accuracy = 100.0 * correct / len(val_loader.dataset)

    return test_loss, accuracy, iou, dice, precision, recall, f1