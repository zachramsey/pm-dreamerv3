import numpy as np
import torch
import torch.nn.functional as F

class TradingEnv:
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.init_cash = cfg["init_cash"]
        self.rew_scale = cfg["reward_scale"]
        self.num_stocks = None
        self.num_features = None

        self.curr_step = 0
        self.value = None
        self.holdings = None
        self.cash = None
        self.history = None
        self.returns = None

    def reset(self, features):
        self.num_stocks, self.num_features = features.shape[-2:]

        self.curr_step = 0
        self.value = torch.tensor(self.init_cash).to(self.device)
        self.holdings = torch.zeros(self.num_stocks).to(self.device)
        self.cash = self.init_cash
        self.history = [(self.value, self.holdings, self.cash)]
        self.returns = torch.tensor([]).to(self.device)

        actions = torch.zeros(self.num_stocks).to(self.device)
        state = self._get_state(actions, features)

        return state

    def step(self, actions, features, targets):
        self.curr_step += 1
        
        self._execute_trades(actions, targets)          # Execute the trades based on the actions
        next_state = self._get_state(actions, features) # Get the next state
        reward = self._get_reward()                     # Get the reward
        
        value = self.value.cpu().numpy()
        done = value <= self.init_cash * 0.1           # Check if the value is depleted

        return next_state, reward, done
    
    def _execute_trades(self, actions, targets):
        actions = F.softmax(actions, dim=0).flatten()
        targets = targets
        
        prev_val = self.value
        curr_val = self.cash + torch.dot(targets, self.holdings)

        self.holdings = torch.floor((curr_val * actions) / targets)
        stock_val = torch.dot(targets, self.holdings)

        self.cash = curr_val - stock_val
        self.value = stock_val + self.cash

        log_return = torch.log(self.value / prev_val)
        self.returns = torch.cat([self.returns, log_return.unsqueeze(0)])
        self.history.append((self.value, self.holdings, self.cash))
    
    def _get_state(self, actions, features):
        state = torch.cat([features, actions.view(-1, 1)], dim=1)   # (num_stocks, num_features + 1)
        return state
    
    def _get_reward(self):
        # Last log return
        return self.returns[-1].cpu().item() * self.rew_scale
    
        # Trade-Level Sharpe Ratio
        # num_days = len(self.returns)
        # mean_return = torch.mean(self.returns).cpu().item()
        # std_return = torch.std(self.returns).cpu().item()
        # sharpe = np.sqrt(num_days) * mean_return / std_return if std_return > 0 else 0
        # return sharpe * self.rew_scale
