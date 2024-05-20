import torch
from torch.autograd import grad
import torch.nn as nn


def EYE(r, x):
    """
    Args:
        r (torch.Tensor): Risk factors indicator, should be the same shape as `x`.
        x (torch.Tensor): Attribution tensor, typically some model outputs or activations.

    Returns:
        torch.Tensor: Calculated regularization term.

    Raises:
        ValueError: If the shapes of `r` and `x` do not match.
    """
    if r.shape[-1] != x.shape[-1]:
        raise ValueError("The shapes of risk factors 'r' and attributions 'x' must match for EYE computation.")
    
    l1 = (x * (1 - r)).abs().sum(-1)
    l2sq = ((r * x) ** 2).sum(-1)
    return l1 + torch.sqrt(l1 ** 2 + l2sq)

# # increase reg strength
# def EYE(r, x, beta=5.0):
#     # Beta could be a parameter that modifies how the EYE function penalizes
#     l1 = beta * (x * (1 - r)).abs().sum(-1)
#     l2sq = beta * ((r * x) ** 2).sum(-1)
#     return l1 + torch.sqrt(l1 ** 2 + l2sq)


class cbm_loss(nn.Module):
    def __init__(self, lambda_concept=1):
        super(cbm_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_concept = lambda_concept

    def forward(self, y_pred, y_label):
        concepts_pred, y_pred = y_pred
        concepts_label, y_label = y_label[:, :-1], y_label[:, -1]
        return self.mse_loss(y_pred, y_label) + self.lambda_concept * self.mse_loss(concepts_pred, concepts_label)

