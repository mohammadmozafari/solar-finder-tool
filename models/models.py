import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from torchvision.models import resnet34

class DinoClassifier(nn.Module):

    def __init__(self, layers=1, mode='small', stacked_layers=1) -> None:
        super().__init__()
        
        self.layers = layers
        self.stacked_layers = stacked_layers
        
        if mode == 'small':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            if stacked_layers == 1:
                self.linear_head = nn.Linear((1 + layers) * 384, 2)
            elif stacked_layers == 2:
                self.linear_head = nn.Linear((1 + layers) * 384, 100)
                self.linear_head2 = nn.Linear(100, 2)
        
        elif mode == 'base':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            if stacked_layers == 1:
                self.linear_head = nn.Linear((1 + layers) * 768, 2)
            elif stacked_layers == 2:
                self.linear_head = nn.Linear((1 + layers) * 768, 200)
                self.linear_head2 = nn.Linear(200, 2)
            
        elif mode == 'giant':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            if stacked_layers == 1:
                self.linear_head = nn.Linear((1 + layers) * 1536, 2)
            elif stacked_layers == 2:
                self.linear_head = nn.Linear((1 + layers) * 1536, 300)
                self.linear_head2 = nn.Linear(300, 2)

    def forward(self, x):
        
        if self.layers == 1:
            x = self.dinov2.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.dinov2.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        
        x = self.linear_head(linear_input)
        if self.stacked_layers == 2: x = F.relu(self.linear_head2(x))
        
        return x

class ResnetBase(nn.Module):
    """ResNet pretrained on Imagenet. This serves as the
    base for the classifier, and subsequently the segmentation model

    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """
    def __init__(self, imagenet_base: bool = True) -> None:
        super().__init__()

        resnet = resnet34(pretrained=imagenet_base).float()
        self.pretrained = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # Since this is just a base, forward() shouldn't directly
        # be called on it.
        raise NotImplementedError

class Classifier(ResnetBase):
    """A ResNet34 Model

    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, imagenet_base: bool = True) -> None:
        super().__init__(imagenet_base=imagenet_base)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0), -1))

class Segmenter(ResnetBase):
    """A ResNet34 U-Net model, as described in
    https://github.com/fastai/fastai/blob/master/courses/dl2/carvana-unet-lrg.ipynb

    Attributes:
        imagenet_base: boolean, default: False
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, imagenet_base: bool = False) -> None:
        super().__init__(imagenet_base=imagenet_base)

        self.target_modules = [str(x) for x in [2, 4, 5, 6]]
        self.hooks = self.add_hooks()

        self.relu = nn.ReLU()
        self.upsamples = nn.ModuleList([
            UpBlock(512, 256, 256),
            UpBlock(256, 128, 256),
            UpBlock(256, 64, 256),
            UpBlock(256, 64, 256),
            UpBlock(256, 3, 16),
        ])
        self.conv_transpose = nn.ConvTranspose2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def add_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        hooks = []
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                hooks.append(child.register_forward_hook(self.save_output))
        return hooks

    def retrieve_hooked_outputs(self) -> List[torch.Tensor]:
        # to be called in the forward pass, this method returns the tensors
        # which were saved by the forward hooks
        outputs = []
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                outputs.append(child.output)
        return outputs

    def cleanup(self) -> None:
        # removes the hooks, and the tensors which were added
        for name, child in self.pretrained.named_children():
            if name in self.target_modules:
                # allows the method to be safely called even if
                # the hooks aren't there
                try: del child.output
                except AttributeError: continue
        for hook in self.hooks: hook.remove()

    @staticmethod
    def save_output(module, input, output):
        # the hook to add to the target modules
        module.output = output

    def load_base(self, state_dict: dict) -> None:
        # This allows a model trained on the classifier to be loaded
        # into the model used for segmentation, even though their state_dicts
        # differ
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        org_input = x
        x = self.relu(self.pretrained(x))
        # we reverse the outputs so that the smallest output
        # is the first one we get, and the largest the last
        interim = self.retrieve_hooked_outputs()[::-1]

        for upsampler, interim_output in zip(self.upsamples[:-1], interim):
            x = upsampler(x, interim_output)
        x = self.upsamples[-1](x, org_input)
        return self.sigmoid(self.conv_transpose(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, across_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        up_out = across_out = out_channels // 2
        self.conv_across = nn.Conv2d(across_channels, across_out, 1)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, up_out, 2, stride=2)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x_up, x_across):
        joint = torch.cat((self.conv_transpose(x_up), self.conv_across(x_across)), dim=1)
        return self.batchnorm(self.relu(joint))
