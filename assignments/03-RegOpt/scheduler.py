from typing import List
import math
from functools import partial

from torch.optim.lr_scheduler import _LRScheduler


def cosine_annealing_warm_restarts(
    t: int, eta_min: float, eta_max: float, t_0: int, t_mult: int
) -> float:
    """
    Closed form learning rate under cosine annealing with warm restarts.

    Within each period, the learning rate gradually decreases
    from eta_max (inclusive) to eta_min (not inclusive)
    like how the cosine function decreases from x=0 to x=pi.
    When the period ends, the learning rate restarts at eta_max.

    This

    Parameters
    ----------
    t : int
        Current epoch number.

        Must be a non-negative integer (>= 0).

    eta_min : float
        The minimum learning rate, never actually taken,
        so you can safely set it to 0.

        When the period gets larger, the actual minimum learning rate
        will approach eta_min.

        Must be non-negative (>=0.0).

    eta_max : float
        The maximum learning rate.

        Must be greater than eta_min (>eta_min).

    t_0 : int
        The initial period length (interval between restarts).

        Must be a positive integer (>=1).

    t_mult : int
        The multiplicative factor of next period length divided by
        this period length.

        If t_mult = 2, for example, the period length will double
        for every warm restart.

        Must be a positive integer (>=1).

    Other Parameters
    ----------------
    t_i : int
        Length of the current period; intermediate variable.

    t_curr : int
        Number of epochs since the last restart; intermediate variable.

        Must satisfy `0 <= t_curr < t_i`.

    Returns
    -------
    lr : float
        The learning rate at this epoch.
    """
    if t_mult == 1:
        t_i = t_0  # length of the current period (between 2 restarts)
        t_curr = t % t_i  # number of epochs since last restart
        # a side effect of modulus: t_curr < t_i strictly holds
        # so eta_min is never achievable; we can safely set eta_min=0
    else:  # t_mult > 1
        i = math.floor(math.log(t / t_0 * (t_mult - 1) + 1, t_mult))
        t_curr = t - t_0 * (t_mult**i - 1) / (t_mult - 1)
        t_i = t_0 * t_mult**i
    lr = eta_min + (eta_max - eta_min) / 2 * (1 + math.cos(math.pi * t_curr / t_i))
    return lr


def visualize_cosine_annealing() -> None:
    """
    Visualization script for cosine annealing with warm restarts
    """
    import numpy as np
    from matplotlib import pyplot as plt

    f1 = partial(
        cosine_annealing_warm_restarts, eta_max=1.0, eta_min=0.0, t_0=1, t_mult=2
    )

    max_t = 600
    t_list = np.linspace(0, max_t, max_t + 1)
    lr_list = np.array([f1(t=t) for t in t_list])
    print(min(lr_list), max(lr_list))
    plt.scatter(t_list, lr_list)
    plt.show()


class CustomLRScheduler(_LRScheduler):
    """
    Custom LR Scheduler using closed-form cosine annealing
    with warm restarts.
    """

    def __init__(self, optimizer, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        Notes by Tianyi:
        This scheduler uses cosine annealing with warm restarts.
        To make it meaningfully different from PyTorch's optim.CosineAnnealingWarmRestarts,
        I implement the learning rate computation in closed form,
        with only the epoch number and the initial hyperparameters as inputs.
        This allows us to only modify the get_lr method,
        without keeping track of internal variables for t_curr and t_i,
        unlike PyTorch's CosineAnnealingWarmRestarts, which modifies step() method too.
        """
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        # using the best hyperparameters from the paper
        self.t_0 = 10
        self.t_mult = 2
        self.eta_min = 0

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Under the hood, it simply calls `self._get_closed_form_lr`.

        Returns
        -------
        lr_list : List[float]
            list of learning rates for each parameter group in `self.optimizer`.
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [base_lr for base_lr in self.base_lrs]
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> List[float]:
        """
        Get the closed form of the current learning rates.

        Returns
        -------
        lr_list : List[float]
            list of learning rates for each parameter group in `self.optimizer`.

        """
        f = partial(
            cosine_annealing_warm_restarts,
            t=self.last_epoch,
            eta_min=self.eta_min,
            t_0=self.t_0,
            t_mult=self.t_mult,
        )
        return [f(eta_max=base_lr) for base_lr in self.base_lrs]
