import math
import torch.nn as nn
import torch


class GELU(nn.Module):
    """
    GELU Activation function
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))