import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from .image_dataset import ImageDataset


def visualize_random_dataset():
    data_transform = transforms.Compose([
        transforms.Resize(size=(244, 244)),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(target_dir=".dataset/model_input/train", transform=data_transform)

    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_dataset), size=[1]).item()
        image, label_idx = train_dataset[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"{train_dataset.classes[label_idx]} - {image.shape[1]}x{image.shape[2]}")
        plt.axis(False)
    plt.show()


def visualize_history(train_history_save_path, test_history_save_path):
    # load saved history
    train_history = pd.read_csv(train_history_save_path)
    test_history = pd.read_csv(test_history_save_path)

    # visualize loss
    plt.plot(train_history["train_epoch"], train_history["train_loss"], label='Training Loss', color='blue')
    plt.plot(test_history["test_epoch"], test_history["test_loss"], label='Testing Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.show()

    # visualize accuracy
    plt.plot(train_history["train_epoch"], train_history["train_accuracy"], label='Training Accuracy', color='blue')
    plt.plot(test_history["test_epoch"], test_history["test_accuracy"], label='Testing Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.legend()
    plt.show()


def visualize_predictions(classes, test_samples, test_labels, pred_labels, model_results):
    plt.figure(figsize=(9, 9))
    rows, cols = 3, 3
    for i, sample in enumerate(test_samples):

        plt.subplot(rows, cols, i + 1)

        plt.imshow(sample.permute(1, 2, 0))

        pred_label = classes[pred_labels[i]]
        truth_label = classes[test_labels[i]]

        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")

        plt.axis(False)

    model_name = model_results["model_name"]
    model_loss = model_results["model_loss"]
    model_accuracy = model_results["model_accuracy"]
    fig_title = f"Model: {model_name} - Loss: {model_loss:0.2f} - Accuracy: %{model_accuracy:0.2f}"
    plt.suptitle(fig_title, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()
