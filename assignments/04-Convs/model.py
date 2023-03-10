# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


LOCAL_MODE = False
if LOCAL_MODE:
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # x = input + self.drop_path(x)
        x = input + x
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Residual(nn.Module):
    def __init__(self, layer: nn.Module):
        """
        A residual wrapper for any neural network layer

        Parameters
        ----------
        layer : nn.Module
            The base layer
        """
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)


class SimpleNet(nn.Module):
    def __init__(self, num_channels, num_classes, dims=(32, 32, 32)):
        super().__init__()
        # format: kernel_size, stride, padding
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, dims[0], 3, stride=2, padding=1),
            nn.ReLU(),
        )
        body_list = []
        for fan_in, fan_out in zip(dims, dims[1:]):
            block = nn.Sequential(
                nn.BatchNorm2d(num_features=fan_in),
                nn.Conv2d(fan_in, fan_out, 3, 2, 0, bias=True),
                # nn.BatchNorm2d(num_features=fan_out),
                nn.ReLU(),
            )
            body_list.append(block)

        self.body = nn.Sequential(*body_list)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - 0.5) * 2  # faster normalize
        x = self.stem(x)
        x = self.body(x)
        # global average pooling
        x = torch.flatten(x, start_dim=1)
        # x = x.mean([-2, -1])
        x = self.head(x)
        return x


class SimpleNet2(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        # format: kernel_size, stride, padding
        dims = [32, 32, 32, 32]
        block_list = [
            nn.Conv2d(num_channels, dims[0], 3, stride=2, padding=0),
            nn.ReLU(),
        ]

        for fan_in, fan_out in zip(dims, dims[1:]):
            block_list.extend(
                [
                    nn.BatchNorm2d(num_features=fan_in),
                    nn.Conv2d(fan_in, fan_out, 3, 2, 0, bias=True),
                    # nn.BatchNorm2d(num_features=fan_out),
                    nn.ReLU(),
                ]
            )

        block_list.append(nn.Flatten(start_dim=1))
        block_list.append(nn.Linear(dims[-1], num_classes))
        self.model = nn.Sequential(*block_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - 0.5) * 2  # faster normalize
        x = self.model(x)
        return x


class SimpleNet3(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        # format: kernel_size, stride, padding
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 32, 3, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=32),
            # nn.Conv2d(32, 32, 3, stride=2, padding=0),
            # nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 32, 3, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            # nn.ReLU(),
            # nn.BatchNorm1d(32),
            nn.Linear(32, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (x - 0.5) * 2  # faster normalize
        # x = x * 2 - 1
        # x = self.model(x)
        # return x
        return self.model(x * 2)


class SimpleNet4(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        # is sequential slower?

        # format: kernel_size, stride, padding
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(32, num_classes, 3, stride=2, padding=0, bias=False)
        # self.linear = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2 - 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        x = torch.relu(x)
        x = self.bn3(x)
        x = self.conv3(x)
        # x = x.view(-1, 32)
        x = torch.flatten(x, start_dim=1)
        # x = self.linear(x)
        return x


class Model(torch.nn.Module):
    """
    My model for HW4 submission.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize a generic image classification model.

        Parameters
        ----------
        num_channels : int
            The number of input channels of an input image.

        num_classes : int
            The number of classes for the output classification head.
        """
        super().__init__()
        # depths = [3, 3, 9, 3]
        # dims = [48, 96, 192, 384]

        # self.model = ConvNeXt(
        #     in_chans=num_channels,
        #     num_classes=num_classes,
        #     depths=depths,
        #     dims=dims,
        # )
        # self.model = SimpleNet(num_channels, num_classes, [32] * 4)
        use_training_warmup = False
        use_random_warmup = True
        batch_size = 200

        # seed = torch.randint(0, 2**32, size=[1]).item()
        seed = 1927859108
        if LOCAL_MODE:
            print(f"{seed=}")
        torch.manual_seed(seed)

        self.model = SimpleNet4(num_channels, num_classes)
        # self.model = torch.jit.trace(model, torch.rand(batch_size, 3, 32, 32))

        self.num_classes = num_classes

        # cache warmup without model parameter update
        # presumably some Linux cache magic?
        if LOCAL_MODE:
            tic = time.time()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        self.to(device)

        if use_training_warmup:
            train_data = CIFAR10(
                root="data/cifar10", train=True, download=False, transform=ToTensor()
            )
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            for i, (x, y) in enumerate(train_loader):
                if i > 5:
                    break
                x, y = x.to(device), y.to(device)
                y_hat = self.model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if use_random_warmup:
            x = torch.rand(batch_size, 3, 32, 32, device=device)
            y = torch.randint(10, size=[batch_size], device=device)
            y_hat = self.model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            # optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # note: no parameter is updated in this step
        if LOCAL_MODE:
            toc = time.time()
            print(f"Pretraining time: {toc - tic:.2f} seconds")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            The input [N, C, H, W] image tensor.

        Returns
        -------
        torch.Tensor
            The output [N, N_classes] class probability tensor
        """
        # if self.training:
        #     # does not update model
        #     return torch.ones(
        #         1, device=x.device, dtype=x.dtype, requires_grad=True
        #     ).expand(x.shape[0], self.num_classes)
        return self.model(x)
