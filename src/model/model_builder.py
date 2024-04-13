from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class SpotTheMaskModel(nn.Module):

    def __init__(self, output_shape):
        super().__init__()

        self.weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=self.weights)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=1280,
                out_features=output_shape,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.model(x)
