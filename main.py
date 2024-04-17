import torch
from torch.backends import mps
from src.data.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights
from src.data.data_visualizer import visualize_random_dataset, visualize_predictions
from src.model.model_builder import SpotTheMaskModel
from src.model.model_trainer import model_trainer, model_evalulater
from src.model.model_utils import save_model
from src.model.model_predictor import make_test_samples, make_predictions

"""device"""
device = "cpu"
if mps.is_available():
    device = "mps"

"""weights"""
weights = EfficientNet_B0_Weights.DEFAULT

"""data"""
train_dataset = ImageDataset(target_dir=".dataset/model_input/train", transform=weights.transforms())
test_dataset = ImageDataset(target_dir=".dataset/model_input/test", transform=weights.transforms())
val_dataset = ImageDataset(target_dir=".dataset/model_input/val", transform=weights.transforms())

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

"""visulize"""
visualize_random_dataset()

"""model"""
model = SpotTheMaskModel(output_shape=len(train_dataset.classes))

"""train and test"""
train_history, test_history = model_trainer(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    output_shape=len(train_dataset.classes),
    device=device,
    epochs=2
)

"""save model"""
save_model(model=model, name="spot_the_mask_model_v0.pth")

"""load model"""
model.load_state_dict(torch.load(".output/models/spot_the_mask_model_v0.pth"))

"""eval"""
model_results = model_evalulater(
    model=model,
    data_loader=test_dataloader,
    output_shape=len(train_dataset.classes),
    device=device,
)

"""predict"""
test_samples, test_labels = make_test_samples()
pred_labels = make_predictions(model=model, test_samples=test_samples, device=device)

"""visualize predictions"""
visualize_predictions(
    classes=test_dataset.classes,
    test_samples=test_samples,
    test_labels=test_labels,
    pred_labels=pred_labels,
    model_results=model_results,
)



