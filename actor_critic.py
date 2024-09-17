import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from utils.networks import ModularMLP
from utils.distributions import SymLogTwoHotDist, ContDist


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.cfg = cfg
        crit_cfg = cfg["critic"]
        input_dim = cfg["recurrent_dim"] + cfg["stochastic_dim"] * cfg["discrete_dim"]
        output_dim = cfg["num_bins"]
        self.model = ModularMLP(input_dim, output_dim, cfg["critic"])

        self.val_norm = Moments(crit_cfg["ema_decay"], crit_cfg["ema_limit"], crit_cfg["ema_low"], crit_cfg["ema_high"])
        self.slow_reg = crit_cfg["slow_regularizer"]

    def forward(self, s):
        s = self.model(s)
        s = SymLogTwoHotDist(s, self.cfg)
        return s

    def loss(self, lam_ret, val_dist, slow_val_dist, weight):
        val_offset, val_scale = self.val_norm(lam_ret)
        lam_ret = (lam_ret - val_offset) / val_scale
        lam_ret = torch.concat([lam_ret, torch.zeros_like(lam_ret[-1:])], dim=0)

        val_loss = val_dist.log_prob(lam_ret.detach()) 
        slow_val_loss = self.slow_reg * val_dist.log_prob(slow_val_dist.mean.detach())
        
        return torch.mean(weight.detach()[:-1] * -(val_loss + slow_val_loss)[:-1])
    

class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()

        actor_cfg = cfg["actor"]

        self.min_std = 0.1
        self.max_std = 1.0
        self.absmax = 1.0

        input_dim = cfg["recurrent_dim"] + cfg["stochastic_dim"] * cfg["discrete_dim"]
        hidden_dim = actor_cfg["hidden_dim"]
        output_dim = 1

        self.model = ModularMLP(input_dim, hidden_dim, actor_cfg)
        self.mean_layer = nn.Linear(hidden_dim, output_dim, device=self.model.device)
        self.mean_layer.apply(self._init_uniform_weight())
        self.std_layer = nn.Linear(hidden_dim, output_dim, device=self.model.device)
        self.std_layer.apply(self._init_uniform_weight())

        self.ret_norm = Moments(actor_cfg["retnorm_decay"], actor_cfg["retnorm_limit"], actor_cfg["retnorm_low"], actor_cfg["retnorm_high"])
        self.adv_norm = Moments(actor_cfg["ema_decay"], actor_cfg["ema_limit"], actor_cfg["ema_low"], actor_cfg["ema_high"])
        self.entropy_reg = actor_cfg["entropy_regularizer"]

    def _init_uniform_weight(self, outscale=1.0):
        def init(m):
            limit = np.sqrt(3 * outscale / ((m.in_features + m.out_features) / 2))
            nn.init.uniform_(m.weight, -limit, limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        return init

    def forward(self, s):
        out = self.model(s)
        mean = self.mean_layer(out)
        std = self.std_layer(out)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
        a_distr = Normal(torch.tanh(mean), std)
        a_distr = ContDist(Independent(a_distr, 1), absmax=self.absmax)
        return a_distr, a_distr.sample()

    def sample(self, s):
        mean = torch.randn(1).to(s.device)
        std = torch.randn(1).to(s.device)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
        a_distr = Normal(torch.tanh(mean), std)
        a_distr = ContDist(Independent(a_distr, 1), absmax=self.absmax)
        return a_distr, a_distr.sample()
    
    def loss(self, lam_ret, targ_val, policy, acts, weight):
        _, ret_scale = self.ret_norm(lam_ret)
        advantage = (lam_ret - targ_val[:-1]) / ret_scale
        adv_offset, adv_scale = self.adv_norm(advantage)
        norm_advantage = (advantage - adv_offset) / adv_scale
        log_policy = policy.log_prob(acts)[:-1]
        entropy = policy.entropy()[:-1]
        return torch.mean(weight[:-1] * -(log_policy * norm_advantage.detach() + self.entropy_reg * entropy))


class Moments(nn.Module):
    def __init__(self, rate=0.01, limit=1e-8, per_low=0.05, per_high=0.95):
        super().__init__()
        self.rate = rate
        self.limit = torch.tensor(limit, dtype=torch.float32)
        self.per_low = per_low
        self.per_high = per_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x):
        x = x.detach()
        if self.per_low > 0.0 or self.per_high < 1.0:
            low = torch.quantile(x, self.per_low, dim=0)
            high = torch.quantile(x, self.per_high, dim=0)
            self.low = (1 - self.rate) * self.low + self.rate * low
            self.high = (1 - self.rate) * self.high + self.rate * high
            offset = self.low
            span = self.high - self.low
            span = torch.max(self.limit, span)
            return offset, span
        else:
            return 0.0, 1.0
        
    def stats(self):
        if self.per_low > 0.0 or self.per_high < 1.0:
            offset = self.low
            span = self.high - self.low
            span = torch.max(self.limit, span)
            return offset, span
        else:
            return 0.0, 1.0
