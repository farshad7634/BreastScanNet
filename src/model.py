import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=3):
    return BreastCancerClassifier(num_classes=num_classes)
class BreastCancerClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(BreastCancerClassifier, self).__init__()

        # Load ResNet18 pretrained on ImageNet
        self.model = models.resnet18(pretrained=True)

        # Modify the first convolution layer to accept 1-channel grayscale input
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  # grayscale
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace the final fully connected layer for our 3 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
