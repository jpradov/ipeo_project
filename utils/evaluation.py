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
    all_labels = []
    all_predictions = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # Assuming this is a pixel-wise classification

        correct += pred.eq(target.view_as(pred)).sum().item()
        all_labels.extend(target.view(-1).cpu().numpy())
        all_predictions.extend(pred.view(-1).cpu().numpy())
        test_loss += criterion(output, target).item() * len(data)

    test_loss /= len(val_loader.dataset)

    # Calculate Metrics
    intersection = np.logical_and(all_labels, all_predictions)
    union = np.logical_or(all_labels, all_predictions)
    iou = intersection.sum()/union.sum()
    dice = 2.0 * intersection.sum() / (all_labels.sum() + all_predictions.sum())
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    accuracy = 100.0 * correct / len(val_loader.dataset)

    return test_loss, accuracy, iou, dice, precision, recall, f1