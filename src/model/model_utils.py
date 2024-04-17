import os
import torch
from pathlib import Path
import pandas as pd


def save_model(model, name):
    model_path = Path(".output/models")
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = f"{model_path}/{name}"

    print(f"Saving the model to {model_save_path}")

    torch.save(obj=model.state_dict(), f=model_save_path)


def save_train_history(train_history):
    results_path = Path(".output/results")
    results_path.mkdir(parents=True, exist_ok=True)

    train_history_save_path = results_path / "train_history.csv"

    train_losses = []
    train_accuracies = []
    train_epochs = []

    for train_item in train_history:
        train_losses.append(train_item["train_loss"])
        train_accuracies.append(train_item["train_accuracy"])
        train_epochs.append(train_item["train_epoch"])

    data = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'train_epoch': train_epochs,
    }
    df = pd.DataFrame(data)
    df.to_csv(train_history_save_path, index=False)

    return train_history_save_path


def save_test_history(test_history):

    results_path = Path(".output/results")
    results_path.mkdir(parents=True, exist_ok=True)

    test_history_save_path = results_path / "test_history.csv"

    test_losses = []
    test_accuracies = []
    test_epochs = []

    for test_item in test_history:
        test_losses.append(test_item["test_loss"])
        test_accuracies.append(test_item["test_accuracy"])
        test_epochs.append(test_item["test_epoch"])

    data = {
        'test_loss': test_losses,
        'test_accuracy': test_accuracies,
        'test_epoch': test_epochs,
    }
    df = pd.DataFrame(data)
    df.to_csv(test_history_save_path, index=False)

    return test_history_save_path
