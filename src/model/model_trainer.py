import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from .model_utils import save_model, save_train_history, save_test_history
from ..data.data_visualizer import visualize_history


def model_trainer(model, train_dataloader, test_dataloader, output_shape, device, epochs):
    # model config
    optimizer = SGD(params=model.parameters(), lr=0.1)
    loss_fn = CrossEntropyLoss()
    accuracy_fn = Accuracy(task="multiclass", num_classes=output_shape).to(device)

    train_history = []
    test_history = []

    # model training
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        train_results = train_step(model, train_dataloader, loss_fn, accuracy_fn, optimizer, device)
        test_results = test_step(model, test_dataloader, loss_fn, accuracy_fn, device)

        train_results["train_epoch"] = epoch
        test_results["test_epoch"] = epoch

        train_history.append(train_results)
        test_history.append(test_results)

        print(f"Train Results: {train_results}")
        print(f"Test Results: {test_results}")

    train_history_save_path = save_train_history(train_history)
    test_history_save_path = save_test_history(test_history)

    visualize_history(train_history_save_path, test_history_save_path)

    return train_history, test_history


def model_evalulater(model, data_loader, output_shape, device):
    loss_fn = CrossEntropyLoss()
    accuracy_fn = Accuracy(task="multiclass", num_classes=output_shape).to(device)

    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        eval_loss = 0
        eval_accuracy = 0
        y_preds = []

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            y_logit = model(x)
            eval_loss += loss_fn(y_logit, y)

            y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
            y_preds.append(y_pred.cpu())
            eval_accuracy += accuracy_fn(y_pred, y)

        eval_loss /= len(data_loader)
        eval_accuracy /= len(data_loader)

    model_results = {
        "model_name": model.__class__.__name__,
        "model_loss": eval_loss.item(),
        "model_accuracy": eval_accuracy.item() * 100,
        "model_preds": y_preds
    }

    return model_results


def train_step(model, train_dataloader, loss_fn, accuracy_fn, optimizer, device):
    model.train()
    model.to(device)

    train_loss = 0
    train_accuracy = 0

    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)

        y_logit = model(x)
        loss = loss_fn(y_logit, y)
        train_loss += loss

        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        train_accuracy += accuracy_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"You looked at {batch}/{len(train_dataloader)} samples")

    train_loss /= len(train_dataloader)
    train_accuracy /= len(train_dataloader)

    train_results = {
        "train_loss": train_loss.item(),
        "train_accuracy": train_accuracy.item() * 100,
    }

    return train_results


def test_step(model, test_dataloader, loss_fn, accuracy_fn, device):
    model.eval()
    model.to(device)

    with torch.inference_mode():
        test_loss = 0
        test_accuracy = 0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            y_logit = model(x)
            test_loss += loss_fn(y_logit, y)

            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            test_accuracy += accuracy_fn(y_pred, y)

        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)

        test_results = {
            "test_loss": test_loss.item(),
            "test_accuracy": test_accuracy.item() * 100,
        }

    return test_results
