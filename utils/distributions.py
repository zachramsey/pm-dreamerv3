
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from math import exp
from utils.utils import symlog, symexp


# Adapted from github:Eclectic-Sheep/sheeprl/utils/distribution.py
#         from github:danijar/dreamerv3/jaxutils.py 
class SymLogDist:
    def __init__(self, mode, dims, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self.dims = tuple([-dim for dim in range(1, dims + 1)])
        self.dist = dist
        self.agg = agg
        self.tol = tol

    @property
    def mode(self):
        return symexp(self._mode)

    @property
    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        if self.dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = torch.where(distance < self.tol, 0, distance)
        elif self.dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self.tol, 0, distance)
        else: raise NotImplementedError(self.dist)
        
        if self.agg == "mean": loss = torch.mean(distance, dim=self.dims)
        elif self.agg == "sum": loss = torch.sum(distance, dim=self.dims)
        else: raise NotImplementedError(self.agg)
        
        return -loss
    

class TwoHotDist:
    def __init__(self, logits, cfg):
        self.device = logits.device

        self.logits = logits
        self.probs = torch.softmax(logits, dim=-1)

        self.bins = torch.linspace(cfg["low"], cfg["high"], cfg["num_bins"], device=self.device)

    @property
    def mean(self):
        return torch.sum(self.probs * self.bins, dim=-1)
    
    @property
    def mode(self):
        return torch.sum(self.probs * self.bins, dim=-1)
    
    @property
    def entropy(self):
        return -torch.sum(self.probs * torch.log(self.probs), dim=-1)
    
    def sample(self):
        return torch.sum(F.one_hot(torch.multinomial(self.probs, 1), len(self.bins)) * self.bins, dim=-1)

    def log_prob(self, value):
        below = torch.sum((self.bins <= value[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.bins) - torch.sum((self.bins > value[..., None]).to(torch.int32), dim=-1)

        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)

        _below = torch.where(below == above, 1.0, torch.abs(self.bins[below] - value))
        _above = torch.where(below == above, 1.0, torch.abs(self.bins[above] - value))

        total = _below + _above
        _below = F.one_hot(below, len(self.bins)) * (_below / total)[..., None]
        _above = F.one_hot(above, len(self.bins)) * (_above / total)[..., None]

        return torch.sum((_below + _above) * (self.logits - torch.logsumexp(self.logits, -1, keepdims=True)), dim=-1)
    

class SymLogTwoHotDist:
    def __init__(self, logits, cfg):
        self.two_hot_dist = TwoHotDist(logits, cfg)

    @property
    def mean(self):
        return symexp(self.two_hot_dist.mean)
    
    @property
    def mode(self):
        return symexp( self.two_hot_dist.mode)

    def log_prob(self, value):
        return self.two_hot_dist.log_prob(symlog(value))
    

class BernoulliSafeMode(Bernoulli):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)

    @property
    def mode(self):
        return (self.probs > 0.5).to(self.probs)