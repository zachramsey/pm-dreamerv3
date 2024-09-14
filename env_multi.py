import numpy as np
import torch
import torch.nn.functional as F

class TradingEnv:
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.init_cash = cfg["init_cash"]
        self.rew_scale = cfg["reward_scale"]
        self.num_stocks = cfg["obs_dim"]
        self.num_features = cfg["feat_dim"]

        self.curr_step = 0
        self.holdings = None
        self.value = None
        self.history = None
        self.returns = None

    def reset(self, features):
        # features shape: (1, num_stocks, num_features)

        self.curr_step = 0
        self.holdings = torch.zeros(self.num_stocks).to(self.device)
        self.holdings[0] = self.init_cash
        self.value = torch.tensor(self.init_cash).to(self.device)
        self.history = [(self.holdings, self.value)]
        self.returns = torch.tensor([]).to(self.device)

        actions = torch.zeros(self.num_stocks).to(self.device)
        actions[0] = 1
        state = self._get_state(actions, features)

        return state

    def step(self, actions, features, targets):
        # actions shape: (1, num_stocks)
        # features shape: (1, num_stocks, num_features)
        # targets shape: (1, num_stocks)

        self.curr_step += 1                             # Move to the next time step
        
        actions = actions.squeeze(0)
        features = features.squeeze(0)
        targets = targets.squeeze(0)

        self._execute_trades(actions, targets)          # Execute the trades based on the actions
        next_state = self._get_state(actions, features) # Get the next state
        reward = self._get_reward()                     # Get the reward
        
        value = self.value.cpu().numpy()
        done = value <= self.init_cash * 0.1           # Check if the value is depleted

        return next_state, reward, done
    
    def _execute_trades(self, actions, targets):
        # actions shape: (num_stocks)
        # targets shape: (num_stocks)

        actions = F.softmax(actions, dim=0)

        prices = torch.cat([torch.tensor([1]).to(self.device), targets])
        
        prev_val = self.value
        curr_val = torch.dot(prices, self.holdings)

        self.holdings[1:] = torch.floor((curr_val * actions[1:]) / prices[1:])
        stock_val = torch.dot(prices[1:], self.holdings[1:])

        self.holdings[0] = curr_val - stock_val
        self.value = (stock_val + self.holdings[0])

        log_return = torch.log(self.value / prev_val)
        self.returns = torch.cat([self.returns, log_return.unsqueeze(0)])
        self.history.append((self.holdings, self.value))
    
    def _get_state(self, actions, features):
        # actions shape: (num_stocks)
        # features shape: (1, num_stocks, num_features)

        actions = actions.unsqueeze(1)
        features = features.squeeze(0)

        cash_features = torch.zeros(1, self.num_features-1).to(self.device)
        state = torch.cat([cash_features, features], dim=0)
        state = torch.cat([state, actions], dim=1)

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
