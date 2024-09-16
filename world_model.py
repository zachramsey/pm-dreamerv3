
import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough, kl_divergence

from utils.networks import ModularMLP
from utils.distributions import SymLogTwoHotDist, SymLogDist, symlog
from utils.utils import unimix, get_stochastic_state


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super(WorldModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_dim = cfg["batch_dim"]
        self.batch_len = cfg["batch_len"]

        self.pred_scale = cfg["pred_scale"]
        self.dyn_scale = cfg["dyn_scale"]
        self.rep_scale = cfg["rep_scale"]
        self.free_nats = cfg["free_nats"]
        
        self.stoch_dim = cfg["stochastic_dim"]
        self.discr_dim = cfg["discrete_dim"]
        self.stoch_state_dim = self.stoch_dim * self.discr_dim
        self.recur_state_dim = cfg["recurrent_dim"]

        self.sequence_model = SequenceModel(cfg)
        self.encoder = Encoder(cfg)
        self.dynamics_predictor = DynamicsPredictor(cfg)
        self.reward_predictor = RewardPredictor(cfg)
        self.decoder = Decoder(cfg)

        self.init_recur = nn.Parameter(torch.zeros(cfg["recurrent_dim"], device=self.device))

    def initial_state(self, batch_dim):
        return torch.tanh(self.init_recur).unsqueeze(0).expand(*batch_dim, -1)
    
    def forward(self, x, a):
        h = torch.zeros(self.batch_dim, self.recur_state_dim, device=self.device)
        z_stoch = torch.zeros(self.batch_dim, self.stoch_state_dim, device=self.device)

        hs = torch.empty(self.batch_len, self.batch_dim, self.recur_state_dim, device=self.device)
        zs_stoch = torch.empty(self.batch_len, self.batch_dim, self.stoch_state_dim, device=self.device)
        
        zs_distr = torch.empty(self.batch_len, self.batch_dim, self.stoch_state_dim, device=self.device)
        zs_hat_distr = torch.empty(self.batch_len, self.batch_dim, self.stoch_state_dim, device=self.device)

        for t, (x_, a_) in enumerate(zip(x, a)):
            is_first = t == 0
            a_ = (1 - is_first) * a_
            h = is_first * self.initial_state(h.shape[:-1]) + (1 - is_first) * h
            
            z_distr, z_stoch = self.encoder(h, x_)      # Encode the input into the latent state
            h = self.sequence_model(h, z_stoch, a_)     # Generate the next hidden state
            z_hat_distr, _ = self.dynamics_predictor(h) # Predict the latent state

            hs[t], zs_stoch[t], zs_distr[t], zs_hat_distr[t] = h, z_stoch, z_distr, z_hat_distr

        x_hat = self.decoder(torch.cat((hs, zs_stoch), -1))
        r_hat = self.reward_predictor(torch.cat((hs, zs_stoch), -1))

        zs_distr = torch.unflatten(zs_distr, -1, (self.stoch_dim, self.discr_dim))
        zs_hat_distr = torch.unflatten(zs_hat_distr, -1, (self.stoch_dim, self.discr_dim))

        return hs, zs_stoch, zs_distr, zs_hat_distr, x_hat, r_hat
    
    def imagine(self, h, z_hat_stoch, a):
        h = self.sequence_model(h, z_hat_stoch, a)      # Generate the next hidden state
        _, z_hat_stoch = self.dynamics_predictor(h)     # Predict the next latent state
        return h, z_hat_stoch
    
    def loss(self, x, x_hat, r, r_hat, z_distr, z_hat_distr):
        # Prediction loss: trains decoder/reward via symlog squared loss, continue via logistic regression
        l_pred = -(x_hat.log_prob(x) + r_hat.log_prob(r))

        # Dynamics loss: trains the sequence model to predict the next representation
        l_dyn = torch.max(
            kl_divergence(Independent(OneHotCategoricalStraightThrough(logits=z_distr.detach()), 1),
                          Independent(OneHotCategoricalStraightThrough(logits=z_hat_distr), 1)),
            torch.full(z_distr.shape[:2], self.free_nats, device=self.device))

        # Representation loss: trains the representations to become more predictable
        l_rep = torch.max(
            kl_divergence(Independent(OneHotCategoricalStraightThrough(logits=z_distr), 1),
                          Independent(OneHotCategoricalStraightThrough(logits=z_hat_distr.detach()), 1)),
            torch.full(z_distr.shape[:2], self.free_nats, device=self.device))

        # Combine the losses
        l_rec = torch.mean(l_pred * self.pred_scale + l_dyn * self.dyn_scale + l_rep * self.rep_scale)

        info = {'l_pred': torch.sum(l_pred).item(), 'l_dyn': torch.sum(l_dyn).item(), 
                'l_rep': torch.sum(l_rep).item(), 'l_rec': torch.sum(l_rec).item()}

        return l_rec, info
    

# V2: Representation Model | V3: Encoder
class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Linear(cfg["feat_dim"], cfg["stochastic_dim"], device=self.device)

        input_dim = cfg["recurrent_dim"] + cfg["stochastic_dim"]
        output_dim = cfg["stochastic_dim"] * cfg["discrete_dim"]
        self.encode_model = ModularMLP(input_dim, output_dim, cfg["encoder"])

        self.discrete_dim = cfg["discrete_dim"]
        self.unimix = cfg["unimix"]

    def forward(self, h, x):
        x = self.embedding(symlog(x))
        x = torch.sigmoid(self.encode_model(torch.cat([h, x], dim=-1)))
        z_distr = unimix(x, self.discrete_dim, self.unimix)
        z_stoch = get_stochastic_state(z_distr, self.discrete_dim)
        return z_distr, z_stoch
    

# V2: Recurrent Model | V3: Sequence Model
class SequenceModel(nn.Module):
    def __init__(self, cfg):
        super(SequenceModel, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = cfg["stochastic_dim"] * cfg["discrete_dim"] + cfg["actor"]["num_bins"]
        recur_dim = cfg["recurrent_dim"]
        self.embedding = nn.Linear(input_dim, recur_dim, device=self.device)
        self.rnn = nn.GRUCell(recur_dim, recur_dim, device=self.device)

    def forward(self, h, z, a):
        x = self.embedding(torch.cat([z, a, h], dim=-1))
        h = self.rnn(x, h)
        return h
        

# V2: Transition Predictor | V3: Dynamics Predictor
class DynamicsPredictor(nn.Module):
    def __init__(self, cfg):
        super(DynamicsPredictor, self).__init__()

        input_dim = cfg["recurrent_dim"]
        output_dim = cfg["discrete_dim"] * cfg["stochastic_dim"]
        self.model = ModularMLP(input_dim, output_dim, cfg["dynamics_predictor"])

        self.discrete_dim = cfg["discrete_dim"]
        self.unimix = cfg["unimix"]

    def forward(self, h, take_sample=True):
        z_hat_distr = unimix(self.model(h), self.discrete_dim, self.unimix)
        z_hat_stoch = get_stochastic_state(z_hat_distr, self.discrete_dim, take_sample)
        return z_hat_distr, z_hat_stoch
    

# V2/V3: Reward Predictor
class RewardPredictor(nn.Module):
    def __init__(self, cfg):
        super(RewardPredictor, self).__init__()
        
        input_dim = cfg["recurrent_dim"] + cfg["stochastic_dim"] * cfg["discrete_dim"]
        output_dim = cfg["num_bins"]
        self.model = ModularMLP(input_dim, output_dim, cfg["reward_predictor"])

        self.cfg = cfg

    def forward(self, s):
        return SymLogTwoHotDist(self.model(s), self.cfg)


# V2: Image Predictor | V3: Decoder
class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        input_dim = cfg["recurrent_dim"] + cfg["stochastic_dim"] * cfg["discrete_dim"]
        output_dim = cfg["feat_dim"]
        self.model = ModularMLP(input_dim, output_dim, cfg["decoder"])

    def forward(self, s):
        return SymLogDist(self.model(s), dims=1)
    

# V2: Discount Predictor | V3: Continue Predictor
# NOTE: My implementation has a deterministic termination signal, 
#       so the continue predictor is not used for training speed.
# class ContinuePredictor(nn.Module):
#     def __init__(self, cfg):
#         super(ContinuePredictor, self).__init__()
#         self.model = ModularMLP(cfg)

#     def forward(self, s):
#         return self.model(s)