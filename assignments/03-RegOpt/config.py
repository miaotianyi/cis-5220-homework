from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
# from torchvision.transforms import RandomHorizontalFlip


class CONFIG:
    batch_size = 32
    num_epochs = 10
    initial_learning_rate = 0.01
    initial_weight_decay = 0.001

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.SGD(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
            # ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # RandomHorizontalFlip(p=0.5),
        ]
    )


# Gaussian noise(0, 0.1): 0.7/0.66
# no noise: 0.7/0.66
# rotate 30: 0.6/0.58
# rotate 10: 0.66/0.63
# horizontal flip p = 0.3: (~85 it/s) 0.69/0.64
# normalize 0.5/0.5, horizontal flip: (~85 it/s) 0.72/0.68
# normalize std=0.3, horizontal flip: (~85 it/s) 0.72/0.68
# flip first, then to tensor: (~85 it/s): 0.72/0.69
# normalize 0.5/0.5: 0.724/0.683
# flip first: 0.721/0.679
# std=1: 0.702/0.661
# color jitter: (~40 it/s): 0.713/0.677
