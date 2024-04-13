import torch
import random
from torchvision import transforms
from ..data.image_dataset import ImageDataset


def make_test_samples():
    data_transform = transforms.Compose([
        transforms.Resize(size=(244, 244)),
        transforms.ToTensor()
    ])

    test_dataset = ImageDataset(target_dir=".dataset/images/test", transform=data_transform)

    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_dataset), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    return test_samples, test_labels


def make_predictions(model, test_samples, device):
    preds_props = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in test_samples:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            y_logits = model(sample)
            y_pred_prop = torch.softmax(y_logits.squeeze(), dim=0)
            preds_props.append(y_pred_prop.cpu())

    pred_labels = torch.stack(preds_props).argmax(dim=1)

    return pred_labels
