
import copy
import torch

from nfnets.agc import AGC
from utils.laprop import LaProp

from world_model import WorldModel
from actor_critic import Actor, Critic

from utils.replay_buffer import ReplayBuffer


class DreamerV3:
    def __init__(self, train_dataloader, eval_dataloader, environment, cfg):
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)

        self.train_dl = train_dataloader
        self.eval_dl = eval_dataloader
        self.env = environment
        self.cfg = cfg

        # Dimensions
        self.batch_dim = cfg["batch_dim"]
        self.batch_len = cfg["batch_len"]

        self.train_len = cfg["train_len"]
        self.eval_len = cfg["test_len"]
        self.obs_dim = cfg["obs_dim"]
        self.feat_dim = cfg["feat_dim"]
        self.act_dim = cfg["act_dim"]

        self.recur_state_dim = cfg["recurrent_dim"]
        self.stoch_state_dim = cfg["stochastic_dim"] * cfg["discrete_dim"]
        self.latent_state_dim = self.recur_state_dim + self.stoch_state_dim

        # Hyperparameters
        self.horizon = cfg["horizon"]
        self.lam = cfg["lam"]
        self.gamma = cfg["gamma"]

        self.target_update_interval = cfg["critic"]["target_update_interval"]
        self.slow_target_frac = cfg["critic"]["slow_target_frac"]

        # Models
        self.world_model = WorldModel(cfg)
        self.actor = Actor(cfg)
        self.critic = Critic(cfg)
        self.critic_target = copy.deepcopy(self.critic)

        self.updates = 0
        
        # Optimizers
        # https://github.com/Z-T-WANG/LaProp-Optimizer
        wm_opt = LaProp(self.world_model.parameters(), lr=cfg["model_lr"], betas=(0.9,0.99), eps=cfg["epsilon"])
        act_opt = LaProp(self.actor.parameters(), lr=cfg["actor_lr"], betas=(0.9,0.99), eps=cfg["epsilon"])
        crit_opt = LaProp(self.critic.parameters(), lr=cfg["critic_lr"], betas=(0.9,0.99), eps=cfg["epsilon"])
        # https://github.com/vballoli/nfnets-pytorch
        self.world_model_optimizer = AGC(self.world_model.parameters(), optim=wm_opt, clipping=cfg["clip_grad"])
        self.actor_optimizer = AGC(self.actor.parameters(), optim=act_opt, clipping=cfg["clip_grad"])
        self.critic_optimizer = AGC(self.critic.parameters(), optim=crit_opt, clipping=cfg["clip_grad"])

        # Experience replay
        self.buffer = ReplayBuffer(cfg)

    def train(self):
        with torch.no_grad():
            self._env_interaction(seed=True)

        for i in range(self.cfg["train_iterations"]):
            print(f"Iteration {i+1}/{self.cfg['train_iterations']}:")

            for c in range(self.cfg["collect_interval"]):
                print(f"\tCollecting data ({c+1}/{self.cfg['collect_interval']})", end="\r")
                
                data = self.buffer.sample()
                x, acts, rew = data["x"], data["a"], data["r"]

                h, z_stoch, loss_info = self._dynamics_learning(x, acts, rew)
                print(f"\tCollecting data ({c+1}/{self.cfg['collect_interval']}) | Reconstruction: {loss_info['l_rec']:.4f}", end="\r")

                actor_loss, critic_loss = self._behavior_learning(h, z_stoch)
                print(f"\tCollecting data ({c+1}/{self.cfg['collect_interval']}) | Reconstruction: {loss_info['l_rec']:.4f} | Actor: {actor_loss:.4f} | Critic: {critic_loss:.4f}", end="\r")
            print()

            with torch.no_grad():
                self._env_interaction()
                self._env_eval()


    def _dynamics_learning(self, x, acts, rew):
        self.world_model_optimizer.zero_grad()
        
        h, z_stoch, z_distr, z_hat_distr, x_hat, r_hat = self.world_model(x, acts)
        rec_loss, loss_info = self.world_model.loss(x, x_hat, rew, r_hat, z_distr, z_hat_distr)

        rec_loss.backward()
        self.world_model_optimizer.step()

        return h, z_stoch, loss_info
    

    def _behavior_learning(self, h, z_hat_stoch):
        if self.updates % self.target_update_interval == 0:
            mix = self.slow_target_frac
            for p, slow_p in zip(self.critic.parameters(), self.critic_target.parameters()):
                slow_p.data = mix * p.data + (1 - mix) * slow_p.data
        self.updates += 1

        h = h.detach().view(-1, self.recur_state_dim)
        z_hat_stoch = z_hat_stoch.detach().view(-1, self.stoch_state_dim)
        s = torch.cat((h, z_hat_stoch), -1)

        traj = torch.empty(self.horizon+1, self.batch_dim*self.batch_len, self.latent_state_dim, device=self.device)
        traj[0] = s

        acts_imag = torch.empty(self.horizon+1, self.batch_dim*self.batch_len, self.act_dim, device=self.device)
        acts = self.actor(s)[1]
        acts_imag[0] = acts

        # Imagine trajectories
        for t in range(1, self.horizon+1):
            h, z_hat_stoch = self.world_model.imagine(h, z_hat_stoch, acts)
            s = torch.cat((h.detach(), z_hat_stoch.detach()), -1)
            traj[t] = s
            acts = self.actor(s)[1]
            acts_imag[t] = acts

        slow_critic = self.critic_target(traj)
        v_offset, v_scale = self.critic_target.val_norm.stats()
        targ_val = slow_critic.mean * v_scale + v_offset
        rew = self.world_model.reward_predictor(traj).mean
        weight = torch.cumprod(torch.cat([torch.ones_like(rew[:1]), self.gamma * torch.ones_like(rew[:-1])], 0), 0).detach()
        
        lam_rets = [targ_val[:-1]]
        discount = torch.ones_like(rew[:-1]) * self.gamma
        interm = rew[1:] + (1 - self.lam) * discount * targ_val[1:]
        for t in reversed(range(discount.shape[0])):
            lam_rets.append(interm[t] + discount[t] * self.lam * lam_rets[-1])
        lam_ret = torch.stack(list(reversed(lam_rets))[:-1], dim=0)

        policy = self.actor(traj.detach())[0]
        actor_loss = self.actor.loss(lam_ret, targ_val, policy, acts_imag, weight)

        val_dist = self.critic(traj[:-1].detach())
        slow_val_dist = self.critic_target(traj[:-1].detach())
        critic_loss = self.critic.loss(lam_ret, val_dist, slow_val_dist, weight)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
    
    def _env_interaction(self, seed=False):
        if seed: episodes = self.cfg["seed_episodes"]
        else: episodes = self.cfg["interaction_episodes"]

        for e in range(episodes):
            with torch.no_grad():
                rand_idx = torch.randint(self.obs_dim-1, (1,)).item()
                
                h = self.world_model.initial_state([1])
                obs = torch.empty(self.train_len, self.feat_dim, device=self.device)
                actions = torch.empty(self.train_len, self.act_dim, device=self.device)
                rewards = torch.empty(self.train_len, device=self.device)

                x, rew, d = None, 0, False
                for t, (features, targets) in enumerate(self.train_dl):
                    features = features[:, rand_idx].view(-1, self.feat_dim-1)
                    targets = targets[:, rand_idx].view(-1)
                    if t == 0:
                        obs[0] = x = self.env.reset(features)
                    else:
                        if seed: 
                            acts = self.actor.sample(x)[1]
                        else:
                            _, z_stoch = self.world_model.encoder(h, x)
                            acts = self.actor(torch.cat((h, z_stoch), -1))[1]
                            h = self.world_model.sequence_model(h, z_stoch, acts)
                        x_next, rew, d = self.env.step(acts, features, targets)
                        obs[t], actions[t], rewards[t] = x.view(obs[t].size()), acts, rew
                        x = x_next

                    print_freq = 250
                    if t % print_freq == 0 or t == self.train_len-1:
                        reward = torch.sum(rewards).item() if t == self.train_len-1 else torch.sum(rewards[t-print_freq:t]).item()
                        profit = self.env.value - self.env.init_cash
                        if seed: print(f"Seeding the replay buffer ({e+1}/{episodes}) | Step {t+1}/{self.train_len}   ", end="\r")
                        else: print(f"\tInteraction | Step {t+1}/{self.train_len} | Reward: {reward:.2f} | Profit: {profit:.2f}", end="\r")

                    if d: break
                self.buffer.extend(obs, actions, rewards)
        print()

    def _env_eval(self):
        with torch.no_grad():
            h = self.world_model.initial_state([self.obs_dim])
            rewards = torch.empty(self.eval_len, device=self.device)
            rewards[0] = 0

            x, d = None, False
            for t, (features, targets) in enumerate(self.eval_dl):
                features = features.view(self.obs_dim, self.feat_dim-1)
                targets = targets.view(self.obs_dim)
                if t == 0: 
                    x = self.env.reset(features)
                else:
                    _, z_stoch = self.world_model.encoder(h, x)
                    acts = self.actor(torch.cat((h, z_stoch), -1))[1]
                    h = self.world_model.sequence_model(h, z_stoch, acts)
                    x, rewards[t], d = self.env.step(acts, features, targets)

                print_freq = 100
                if t % print_freq == 0 or t == self.eval_len-1:
                    reward = torch.sum(rewards).item() if t == self.eval_len-1 else torch.sum(rewards[t-print_freq:t]).item()
                    profit = self.env.value - self.env.init_cash
                    print(f"\t Evaluation | Step {t+1}/{self.eval_len} | Reward: {reward:.2f} | Profit: {profit:.2f}", end="\r")

                if d: break
            print()
