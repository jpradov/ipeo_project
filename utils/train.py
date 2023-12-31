import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.notebook import tqdm  # as we run the functions in notebooks only
from utils.evaluation import evaluate
import wandb


class TrainingResult():
    def __init__(
        self,
        train_loss_history: list[float],
        train_acc_history: list[float],
        val_loss_history: list[float],
        val_acc_history: list[float],
        iou_history: list[float],
        precision_history: list[float],
        recall_history: list[float],
        f1_history: list[float]
    ) -> None:
        self.train_loss_history = train_loss_history
        self.train_acc_history = train_acc_history
        self.val_loss_history = val_loss_history
        self.val_acc_history = val_acc_history
        self.iou_history = iou_history
        self.precision_history = precision_history
        self.recall_history = recall_history
        self.f1_history = f1_history


def run_training(
        experiment_name: str,
        model: Module,
        num_epochs: int,
        optimizer,
        criterion,
        train_dl,
        val_dl,
        scheduler,
        lr: float,
        batch_size: int,
        device="cuda:0",
        project_name=None,
        save=False,
        early_stop_patience=None,
) -> TrainingResult:
    """Function to run the full training of a model over given epoch and logging progress to wandb."""
    
    """`wandb.login()` must be called prior to training"""
    # adapted from CS-433 Machine Learning Exercises
    
    # ===== Setup =====

    # initialise weights and biases
    wandb.init(
        name=experiment_name,
        entity="ipeo_project",
        project=project_name,
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        }
    )

    # Setup
    model = model.to(device=device)

    # if early stop patience is not set, we do not early stop, 
    # i.e., we simply set early stop patience to num_epochs
    if early_stop_patience == None:
        early_stop_patience = num_epochs
    
    early_stopper = _EarlyStopper(patience=early_stop_patience)

    # initialise lists to observe training
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    iou_history = []
    precision_history = []
    recall_history = []
    f1_history = []

    # ===== Train Model =====
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = _train_epoch(
            experiment_name=experiment_name,
            model=model,
            device=device,
            train_loader=train_dl,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            criterion=criterion,
            save=save,
        )

        # extend training history
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)

        # calculate validation progress
        val_loss, val_acc, iou, precision, recall, f1 = evaluate(
            model=model,
            device=device,
            val_loader=val_dl,
            criterion=criterion
        )

        # log to wandb
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "learning rate" : current_lr
        })

        # keep track of all training and validation statistics
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        iou_history.append(iou)
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)

        if early_stopper.early_stop(val_acc):
            print(f"Early stopped at epoch {epoch} with val loss {val_loss} and val accuracy {val_acc}.")
            break

    # ===== Plot training and validation curves =====
    n_train = len(train_acc_history)
    t_train = epoch * np.arange(n_train) / n_train
    t_val = np.arange(1, epoch + 1)
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

    # close wandb run-session
    wandb.finish()

    return TrainingResult(
        train_loss_history=train_loss_history,
        train_acc_history=train_acc_history,
        val_loss_history=val_loss_history,
        val_acc_history=val_acc_history,
        iou_history=iou_history,
        precision_history=precision_history,
        recall_history=recall_history,
        f1_history=f1_history,
    )


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
    experiment_name: str,
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler,
    epoch: int,
    criterion,
    device="cuda:0",
    save=False,
) -> tuple[list[float], list[float]]:
    """Function handling training of one epoch."""
    # adapted from CS-433 Machine Learning Exercises
    
    model.train()
    loss_history = []
    accuracy_history = []

    num_minibatches = len(train_loader.torch_loader) # returns the number of batches in the loader

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=num_minibatches):
        loss, accuracy = _train_batch(data=data, target=target, model=model, optimizer=optimizer,
                                      criterion=criterion, device=device)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        wandb.log({
            "loss": loss,
            "accuracy": accuracy,
        })

        if batch_idx % (len(train_loader.torch_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss:0.2e} batch_acc={accuracy:0.3f}"
            )
            if save:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss
                }, f"{experiment_name}.pt")
    if scheduler != None:
      scheduler.step()
    return loss_history, accuracy_history


def _train_batch(data, target, model: Module, optimizer: Optimizer, criterion, device="cuda:0") -> tuple[float, float]:
    """Function to perform forward and backward pass for a given training batch. """

    data, target = data.to(device=device), target.to(device=device)

    # feed image through model and calculate loss
    output = model(data)
    loss = criterion(output, target)

    # gradient step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # get class predictions
    predictions = output.argmax(1)

    # calculate train metrics
    loss = loss.item()
    accuracy = torch.mean((predictions == target).float()).item()

    return loss, accuracy

