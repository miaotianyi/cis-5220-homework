import torch
from torch import nn
from typing import Callable


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation
        self.initializer = initializer

        layer_widths = (
            [self.input_size]
            + [self.hidden_size] * self.hidden_count
            + [self.num_classes]
        )
        layer_list = []
        for fan_in, fan_out in zip(layer_widths, layer_widths[1:]):
            layer_list.append(nn.Linear(fan_in, fan_out, bias=True))
            layer_list.append(self.activation())
        layer_list = layer_list[:-1]  # drop last activation
        self.net = nn.Sequential(*layer_list)
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                self.initializer(m.weight.data)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        return self.net(x)
