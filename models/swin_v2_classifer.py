import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import swin_v2_t
from torchvision.models import resnet34, resnet50, resnet101

class Base(nn.Module):
    def __init__(self, imagenet_base: bool = True) -> None:
        super().__init__()

        # resnet 34, 50, 101
        # resnet 100
        # base = resnet34(pretrained=imagenet_base).float()
        # resnet = resnet50(pretrained=imagenet_base).float()
        # resnet = resnet101(pretrained=imagenet_base).float()
        # self.pretrained = nn.Sequential(*list(base.children())[:-2])
        
        # vision transformers
        base = swin_v2_t(pretrained=imagenet_base).float()
        self.pretrained = nn.Sequential(*list(base.children())[:-3])

class SwinV2Classifier(Base):

    def __init__(self, imagenet_base: bool = True) -> None:
        super().__init__(imagenet_base=imagenet_base)

        # swin_v2_t
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x