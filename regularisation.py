import torch
from torch.autograd import grad

def EYE(r, x):
    """
    Expert Yielded Estimation (EYE) for regularization.

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
