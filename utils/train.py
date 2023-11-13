from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import datasets, transforms
import tqdm
from data import create_dataloaders
from evaluation import evaluate
import config

# TODO: check out other functions from the IPEO deep learning exercises (semantic segmentation and convnets), they might be useful too


def run_training(model: Module, num_epochs: int, lr: float, batch_size: int, num_workers=2, device="cpu"):
    # adapted from CS-433 Machine Learning Exercises
    # ===== Data Loading =====
    train_dl, val_dl, test_dl = create_dataloaders(
        batch_size=batch_size, num_workers=num_workers)

    # ===== Model, Optimizer and Criterion =====
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.functional.cross_entropy

    # ===== Train Model =====
    early_stopper = _EarlyStopper()
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    iou_history = []
    dice_history = []
    precision_history = []
    recall_history = []
    f1_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = _train_epoch(
            model=model,
            device=device,
            train_loader=train_dl,
            optimizer=optimizer,
            epoch=epoch,
            criterion=criterion
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)

        val_loss, val_acc, iou, dice, precision, recall, f1 = evaluate(
            model=model,
            device=device,
            val_loader=val_dl,
            criterion=criterion
        )
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        iou_history.append(iou)
        dice_history.append(dice)
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)
        if early_stopper.early_stop(val_loss):
            break

    # TODO - plot all validation data

    # ===== Plot training curves =====
    n_train = len(train_acc_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_val = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(t_train, train_acc_history, label="Train")
    plt.plot(t_val, val_acc_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.figure()
    plt.plot(t_train, train_loss_history, label="Train")
    plt.plot(t_val, val_loss_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # ===== Plot low/high loss predictions on validation set =====
    points = _get_predictions(
        model=model,
        device=device,
        val_loader=val_dl,
        criterion=partial(torch.nn.functional.cross_entropy, reduction="none"),
    )
    points.sort(key=lambda x: x[1])
    plt.figure(figsize=(15, 6))
    for k in range(5):
        plt.subplot(2, 5, k + 1)
        plt.imshow(points[k][0].reshape(28, 28), cmap="gray")
        plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
        plt.subplot(2, 5, 5 + k + 1)
        plt.imshow(points[-k - 1][0].reshape(28, 28), cmap="gray")
        plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")


def load_checkpoint(checkpoint_path: str, model: Module, optimizer: Module) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["checkpoint_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


class _EarlyStopper:
    # from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def _train_epoch(
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    criterion,
    device="cpu"
) -> tuple[list[float], list[float]]:
    # adapted from CS-433 Machine Learning Exercises
    model.train()
    loss_history = []
    accuracy_history = []
    pbar = tqdm(total=100)
    for batch_idx, (data, target) in enumerate(train_loader):
        loss, accuracy = _train_batch(data=data, target=target, model=model, optimizer=optimizer,
                                      criterion=criterion, device=device)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            pbar.update(10)
            print(
                f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss/len(data):0.2e} batch_acc={accuracy/len(data):0.3f}"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss
            }, config.CHECKPOINT_PATH)

    return loss_history, accuracy_history


def _train_batch(data, target, model: Module, optimizer: Optimizer, criterion, device="cpu") -> tuple[float, float]:
    data, target = data.to(device=device), target.to(device=device)

    output = model.forward(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    predictions = output.argmax(1).cpu().detatch().numpy()
    ground_truth = target.cpu().detatch().numpy()

    loss = loss.item().cpu().detatch().numpy()
    accuracy = (predictions == ground_truth).mean()

    return loss, accuracy


@torch.no_grad()
def _get_predictions(model, device, val_loader, criterion, num=None):
    # adapted from CS-433 Machine Learning Exercises
    model.eval()
    points = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)

        data = np.split(data.cpu().numpy(), len(data))
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points
