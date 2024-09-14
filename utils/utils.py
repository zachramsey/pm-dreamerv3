
import torch

#----------------------------------------------------------------------------------------------------

from torch.distributions import Independent, OneHotCategoricalStraightThrough

def get_stochastic_state(logits, discrete_dim, sample=True):
    """### Returns the stochastic state given the logits
    Args:
        logits (torch.Tensor): The logits.
        discrete_dim (int): The number of discrete dimensions
        sample (bool): Whether to sample the state.
    Returns:
        torch.Tensor: The state.
    """
    logits = torch.unflatten(logits, -1, (-1, discrete_dim))
    logits = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    logits = torch.flatten(logits.rsample() if sample else logits.mode, -2)
    return logits

#----------------------------------------------------------------------------------------------------

from torch.distributions.utils import probs_to_logits

def unimix(logits, discrete_dim, unimix):
    """ Parameterize a distribution as a mixture of the original and a uniform distribution.
    Args:
        logits (torch.Tensor): The logits.
        discrete_dim (int): The number of discrete dimensions.
        unimix (float): The uniform mixing factor.
    Returns:
        torch.Tensor: The logits.
    """
    if logits.dim() == 2:
        logits = logits.view(logits.size(0), -1, discrete_dim)
    assert logits.dim() == 3, f"uniform_mix: logits must have 3 dims, got {logits.dim()} | shape: {logits.shape}"
    if unimix > 0.0:
        logits = torch.softmax(logits, dim=-1)
        uniform = torch.ones_like(logits) / discrete_dim
        logits = (1 - unimix) * logits + unimix * uniform
        logits = probs_to_logits(logits)
    return logits.view(logits.size(0), -1)

#----------------------------------------------------------------------------------------------------

def calc_lam_vals(r, vals, gamma, lam):
    """### Calculate the lambda values
    Args:
        r (torch.Tensor): The rewards.
        vals (torch.Tensor): The values.
        gamma (float): The gamma value.
        lam (float): The lambda value.
    Returns:
        torch.Tensor: The lambda values.
    """
    vals_ = [vals[-1:]]
    interm = r + gamma * vals * (1 - lam)
    for t in reversed(range(len(r))):
        vals_.append(interm[t] + gamma * lam * vals_[-1])
    ret = torch.cat(list(reversed(vals_))[:-1])
    return ret

#----------------------------------------------------------------------------------------------------

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
