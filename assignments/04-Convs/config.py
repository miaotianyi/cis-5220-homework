from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 200
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=5e-3)
    # torch.optim.SGD(model.parameters(), lr=0.1)
    # torch.optim.Adam(model.parameters(), lr=5e-3)

    transforms = Compose([ToTensor()])
    # transforms = Lambda(lambda x: torch.tensor(x))
