from typing import List
import math

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        This scheduler uses cosine annealing with warm restarts.
        To make it meaningfully different from PyTorch's optim.CosineAnnealingWarmRestarts,
        we implement the learning rate computation in closed form,
        with only the epoch number and the initial hyperparameters as inputs.
        This allows us to only modify the get_lr method,
        unlike PyTorch's CosineAnnealingWarmRestarts, which modifies step() method too.
        """
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        # using the best hyperparameters from the paper
        self.T_0 = 10
        self.T_mult = 2
        self.eta_min = 0

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # return [self.eta_min + (eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2 for eta_max in self.base_lrs]

        # Here's our dumb baseline implementation:
        return [base_lr for base_lr in self.base_lrs]
