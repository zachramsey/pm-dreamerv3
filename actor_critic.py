import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils.networks import ModularMLP
from utils.utils import unimix
from utils.distributions import SymLogTwoHotDist, TwoHotDist


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.cfg = cfg
        crit_cfg = cfg["critic"]
        input_dim = cfg["recurrent_dim"] + cfg["stochastic_dim"] * cfg["discrete_dim"]
        output_dim = cfg["num_bins"]
        self.model = ModularMLP(input_dim, output_dim, cfg["critic"])

        self.val_norm = Moments(crit_cfg["ema_decay"], crit_cfg["ema_limit"], crit_cfg["ema_low"], crit_cfg["ema_high"])

    def forward(self, s):
        s = self.model(s)
        s = SymLogTwoHotDist(s, self.cfg)
        return s

    def loss(self, lam_ret, val_dist, slow_val_dist, weight):
        val_offset, val_scale = self.val_norm(lam_ret)
        lam_ret = (lam_ret - val_offset) / val_scale

        lam_ret = torch.concat([lam_ret, torch.zeros_like(lam_ret[-1:])], dim=0)

        val_loss = -val_dist.log_prob(lam_ret.detach()) 
        slow_val_loss = -val_dist.log_prob(slow_val_dist.mean.detach())
        
        return torch.mean(weight.detach()[:-1] * (val_loss + slow_val_loss))
    

class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()

        self.obs_dim = cfg["obs_dim"]

        actor_cfg = cfg["actor"]
        self.unimix = actor_cfg["unimix"]
        self.num_bins = actor_cfg["num_bins"]

        input_dim = cfg["recurrent_dim"] + cfg["stochastic_dim"] * cfg["discrete_dim"]
        output_dim = self.obs_dim #* self.num_bins
        self.model = ModularMLP(input_dim, output_dim, actor_cfg)

        self.ret_norm = Moments(actor_cfg["retnorm_decay"], actor_cfg["retnorm_limit"], actor_cfg["retnorm_low"], actor_cfg["retnorm_high"])
        self.adv_norm = Moments(actor_cfg["ema_decay"], actor_cfg["ema_limit"], actor_cfg["ema_low"], actor_cfg["ema_high"])
        self.entropy_reg = actor_cfg["entropy_regularizer"]

    def forward(self, s, deterministic=True):
        # logits = unimix(self.model(s), self.num_bins, self.unimix)
        # return TwoHotDist(logits, 1, actor_cfg)
        # B = s.shape[:-1]
        a_distr = self.model(s)
        if not deterministic:
            a_distr = torch.clamp(a_distr + torch.randn_like(a_distr) * 0.2, -0.5, 0.5)
        # a_distr = unimix(a_distr, self.num_bins, self.unimix).view(*B, self.obs_dim, self.num_bins)
        # a_distr = TwoHotDist(a_distr, actor_cfg)
        # a_distr = Categorical(logits=a_distr)
        a_distr = torch.softmax(a_distr, dim=-1)
        return a_distr

    def sample(self, s):
        # logits = torch.randn(self.obs_dim, self.num_bins).to(s.device)
        # logits = unimix(logits, self.num_bins, self.unimix)
        # return TwoHotDist(logits.unsqueeze(0), 1, actor_cfg)
        a_distr = torch.randn(1, self.obs_dim).to(s.device)
        # a_distr = unimix(a_distr, self.num_bins, self.unimix).view(1, self.obs_dim, self.num_bins)
        # a_distr = TwoHotDist(a_distr, actor_cfg)
        # a_distr = Categorical(logits=a_distr)
        a_distr = torch.softmax(a_distr, dim=-1)
        return a_distr
    
    def loss(self, lam_ret, targ_val, policy, weight):
        # lam_ret: (15, 1024)
        # tar_val: (16, 1024)
        # policy: (16, 1024, 251)
        # weight: (16, 1024)

        # offset, scale = self.ret_norm(target)
        # target = (target - offset) / scale
        # base = (base[:-1] - offset) / scale
        # advantage = target - base
        # return torch.mean((-weight[:-1] * advantage.detach()) - (self.entropy_reg * policy.entropy()[:-1]))
    
        _, ret_scale = self.ret_norm(lam_ret)
        advantage = (lam_ret - targ_val[:-1]) / ret_scale
        adv_offset, adv_scale = self.adv_norm(advantage)
        norm_advantage = (advantage - adv_offset) / adv_scale
        entropy = self.entropy_reg * policy.entropy()[:-1]
        return torch.mean(weight[:-1] * -(norm_advantage.detach() + entropy))
        # return torch.mean((-weight[:-1] * advantage.detach()) - entropy)
  

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