"""
TODO:
- GRU with block-diagonal recurrent weights of 8 blocks
- Add latents to the replay buffer
- Rework features. The other stuff on yfinance README; test what has historical data
    - EV / EBITDA: (market capitalisation + total debt - cash) / normalized EBITDA
- Seperate the observation dim and action dim so we can have data from sources not being traded
- Reintroduce the continue predictor, make it disablable
- Generalize the model
- Re-evaluate the distributions
- Action repeat
- Make each stock into its own individual thing. the whole iterating over vectors thing
"""

import yaml
from data_loader import StockDataLoader
from environment import TradingEnv
from dreamer import DreamerV3

def main():
    cfg_file = "configs/base.yaml"

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    data_loader = StockDataLoader(cfg)
    train_dl, eval_dl, cfg = data_loader.get_loaders(p_train=0.9)

    env = TradingEnv(cfg)
    dreamer = DreamerV3(train_dl, eval_dl, env, cfg)
    dreamer.train()

if __name__ == "__main__":
    main()
