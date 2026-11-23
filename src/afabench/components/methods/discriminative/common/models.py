# pyright: reportImplicitOverride=false, reportUnannotatedClassAttribute=false
# ruff: noqa: ANN001, ANN003, ANN201, ANN202, B007, C901, D401, EM101, EM102, FBT002, I001, N802, PERF401, PLR0915, PLR1704, PLW2901, PTH118, TRY003, UP008, UP035
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Self, Any, Callable

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.hub import load_state_dict_from_url
from torchrl.modules import MLP

from afabench.components.methods.discriminative.common.utils import (
    restore_parameters,
)
from afabench.components.methods.rl.common.utils import mask_data

models_dir = "./models/pretrained_resnet_models"
model_name = {
    "resnet18": "resnet18-5c106cde.pth",
    "resnet50": "resnet50-19c8e357.pth",
}

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def ResNet18Backbone(model):
    # Remove avgpool and fc
    backbone_modules = nn.ModuleList(model.children())[:-2]
    return nn.Sequential(*backbone_modules), model.expansion


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            ckpt_path = os.path.join(models_dir, model_name["resnet18"])
            state_dict = torch.load(ckpt_path, map_location="cpu")
        except FileNotFoundError:
            state_dict = load_state_dict_from_url(
                model_urls["resnet18"],
                weights_only=False,
            )
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            ckpt_path = os.path.join(models_dir, model_name["resnet50"])
            state_dict = torch.load(ckpt_path, map_location="cpu")
        except FileNotFoundError:
            state_dict = load_state_dict_from_url(
                model_urls["resnet50"],
                weights_only=False,
            )
        model.load_state_dict(state_dict)
    return model


def make_layer(block, in_planes, planes, num_blocks, stride, expansion):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        if stride >= 1:
            downsample = None

            if stride != 1 or in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers.append(block(in_planes, planes, stride, downsample))
        elif expansion == 1:
            layers.append(UpsamplingBlock(in_planes, planes))
        else:
            layers.append(UpsamplingBottleneckBlock(in_planes, planes))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers)


class UpsamplingBlock(nn.Module):
    """Custom residual block for performing upsampling."""

    expansion = 1

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                self.expansion * planes,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.expansion * planes),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class UpsamplingBottleneckBlock(nn.Module):
    """Custom residual block for performing upsampling."""

    expansion = 4

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                planes * self.expansion,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(planes * self.expansion),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class MaskingPretrainer(nn.Module):
    """Pretrain model with missing features."""

    def __init__(self, model, mask_layer):
        super().__init__()
        self.model = model
        self.mask_layer = mask_layer

    def _to_class_indices(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim >= 2:
            return y.argmax(dim=-1).long()
        return y.long()

    def _random_mask(self, x: torch.Tensor, p: float) -> torch.Tensor:
        device = x.device
        if x.dim() != 4:
            _, m, _ = mask_data(x, p)
            return m

        n = self.mask_layer.mask_size
        if n is None:
            return (torch.rand(x.shape, device=device) < p).float()
        return (torch.rand(x.size(0), n, device=device) < p).float()

    def fit(
        self,
        train_loader,
        val_loader,
        lr,
        nepochs,
        loss_fn,
        val_loss_fn=None,
        val_loss_mode=None,
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        early_stopping_epochs=None,
        verbose=True,
        min_mask=0.1,
        max_mask=0.9,
        metric_logger: Callable[[dict[str, float]], None] | None = None,
        metric_prefix: str = "masked_classifier",
    ):
        """Train model."""
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            raise ValueError(
                "must specify val_loss_mode (min or max) when validation_loss_fn is specified"
            )

        # Set up optimizer and lr scheduler.
        model = self.model
        mask_layer = self.mask_layer
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=val_loss_mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        # For tracking best model and early stopping.
        best_model: nn.Module | None = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        for epoch in range(nepochs):
            # Switch model to training mode.
            model.train()
            epoch_train_loss = torch.zeros((), device=device)
            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = self._to_class_indices(y).to(device)

                # Generate missingness.
                p = min_mask + torch.rand(1).item() * (max_mask - min_mask)
                m = self._random_mask(x, p)

                # Calculate loss.
                x_masked = mask_layer(x, m)
                pred = model(x_masked)
                loss = loss_fn(pred, y)

                # Take gradient step.
                loss.backward()
                opt.step()
                model.zero_grad()
                epoch_train_loss += loss.detach()

            avg_train = epoch_train_loss.item() / len(train_loader)

            # Calculate validation loss.
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for x, y in val_loader:
                    # Move to device.
                    x = x.to(device)
                    y = self._to_class_indices(y).to(device)

                    # Generate missingness.
                    p = min_mask + torch.rand(1).item() * (max_mask - min_mask)

                    # Calculate prediction.
                    m = self._random_mask(x, p)
                    x_masked = mask_layer(x, m)
                    pred = model(x_masked)
                    pred_list.append(pred)
                    label_list.append(y)

                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()

            # Print progress.
            if verbose:
                print(f"{'-' * 8}Epoch {epoch + 1}{'-' * 8}")
                print(f"Train loss = {avg_train:.4f}\n")
                print(f"Val loss = {val_loss:.4f}\n")

            if metric_logger is not None:
                metric_logger(
                    {
                        f"{metric_prefix}/epoch": float(epoch + 1),
                        f"{metric_prefix}/train_loss": float(avg_train),
                        f"{metric_prefix}/val_loss": float(val_loss),
                    }
                )

            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_model = deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f"Stopping early at epoch {epoch + 1}")
                break

        # Copy parameters from best model.
        assert best_model is not None
        restore_parameters(model, best_model)

    def forward(self, x, mask):
        """
        Generate model prediction.

        Args:
          x:
          mask:

        """
        x_masked = self.mask_layer(x, mask)
        return self.model(x_masked)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.expansion = block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Classifier(nn.Module):
    def __init__(self, backbone, expansion, num_classes=10):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(512 * expansion, 256, kernel_size=3)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256, num_classes)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(self.bn(self.conv(x)))
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, backbone, expansion=1, block_layer_stride=0.5):
        super(ConvNet, self).__init__()
        self.backbone = backbone
        if expansion == 1:
            # BasicBlock
            self.block_layer = make_layer(
                BasicBlock,
                512,
                256,
                2,
                stride=block_layer_stride,
                expansion=expansion,
            )
            self.conv = nn.Conv2d(256, 1, 1)
        else:
            # Bottleneck block
            # BasicBlock
            self.block_layer = make_layer(
                Bottleneck,
                2048,
                512,
                2,
                stride=block_layer_stride,
                expansion=expansion,
            )
            self.conv = nn.Conv2d(2048, 1, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.backbone(x)
        x = self.block_layer(x)
        # print(x.shape)
        x = self.conv(x)
        x = self.flatten(x)
        return x


def build_mlp(arch: dict[str, Any]) -> nn.Module:
    return MLP(
        in_features=arch["in_features"],
        out_features=arch["out_features"],
        num_cells=arch["hidden_units"],
        activation_class=getattr(nn, arch["activation"]),
        dropout=arch["dropout"],
    )


def build_resnet18_classifier(arch: dict[str, Any]) -> nn.Module:
    base = resnet18(pretrained=bool(arch.get("pretrained", True)))
    backbone, expansion = ResNet18Backbone(base)
    return Classifier(backbone, expansion, num_classes=int(arch["d_out"]))


def build_resnet50_classifier(arch: dict[str, Any]) -> nn.Module:
    base = resnet50(pretrained=bool(arch.get("pretrained", True)))
    backbone, expansion = ResNet18Backbone(base)
    return Classifier(backbone, expansion, num_classes=int(arch["d_out"]))


_BUILDERS: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "mlp": build_mlp,
    "resnet18": build_resnet18_classifier,
    "resnet50": build_resnet50_classifier,
}


class GreedyAFAClassifier:
    def __init__(
        self,
        classifier: nn.Module,
        architecture: dict[str, Any],
        device: torch.device,
    ):
        self.classifier: nn.Module = classifier.to(device)
        self.classifier.eval()
        self.architecture: dict[str, Any] = architecture
        self._device: torch.device = device

    def save(self, path: Path) -> None:
        """Save all necessary files into the given path."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "classifier_state_dict": self.classifier.state_dict(),
                "architecture": self.architecture,
            },
            path / "model.pt",
        )

    @classmethod
    def load(
        cls,
        path: Path,
        device: torch.device,
    ) -> Self:
        checkpoint = torch.load(
            path / "model.pt", map_location=device, weights_only=False
        )
        arch: dict[str, Any] = checkpoint["architecture"]
        state_dict = checkpoint["classifier_state_dict"]
        model_type = arch.get("type")
        if not model_type:
            raise ValueError("Checkpoint missing architecture['type']")

        builder = _BUILDERS.get(str(model_type))
        if builder is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        classifier = builder(arch)
        classifier.load_state_dict(state_dict)
        classifier.eval()

        return cls(classifier=classifier, architecture=arch, device=device)
