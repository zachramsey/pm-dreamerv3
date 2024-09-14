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
from env_multi import TradingEnv
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

"""
# World Model
rssm: 
    deter: 8192
    hidden: 1024
    stoch: 32
    classes: 64
    act: silu
    norm: rms
    unimix: 0.01
    outscale: 1.0
    winit: normal
    imglayers: 2
    obslayers: 1
    dynlayers: 1
    absolute: False
    cell: blockgru
    blocks: 8
    block_fans: False
    block_norm: False
enc: 
    depth: 64
    mults: [1, 2, 3, 4, 4]
    layers: 3
    units: 1024
    act: silu
    norm: rms
    winit: normal
    symlog: True
    outer: True
    kernel: 5
    minres: 4
dec: 
    inputs: [deter, stoch]
    vecdist: symlog_mse
    depth: 64
    mults: [1, 2, 3, 4, 4]
    layers: 3
    units: 1024
    act: silu
    norm: rms
    outscale: 1.0
    winit: normal
    outer: True
    kernel: 5
    minres: 4
    block_space: 8
    block_fans: False
    block_norm: False
    hidden_stoch: True
    space_hidden: 0
rew: 
    layers: 1
    units: 1024
    act: silu
    norm: rms
    dist: symexp_twohot
    outscale: 0.0
    inputs: [deter, stoch]
    winit: normal
    bins: 255
    block_fans: False
    block_norm: False
conhead: 
    layers: 1
    units: 1024
    act: silu
    norm: rms
    dist: binary
    outscale: 1.0
    inputs: [deter, stoch]
    winit: normal
    block_fans: False
    block_norm: False
cont: True

# Actor Critic
actor: 
    layers: 3
    units: 1024
    act: silu
    norm: rms
    minstd: 0.1
    maxstd: 1.0
    outscale: 0.01
    unimix: 0.01
    inputs: [deter, stoch]
    winit: normal
    block_fans: False
    block_norm: False
critic: 
    layers: 3
    units: 1024
    act: silu
    norm: rms
    dist: symexp_twohot
    outscale: 0.0
    inputs: [deter, stoch]
    winit: normal
    bins: 255
    block_fans: False
    block_norm: False

actor_dist_disc: onehot
actor_dist_cont: normal
imag_start: all
imag_repeat: 1
imag_length: 15
imag_unroll: False
horizon: 333
return_lambda: 0.95
return_lambda_replay: 0.95
slow_critic_update: 1
slow_critic_fraction: 0.02
retnorm: 
    impl: perc
    rate: 0.01
    limit: 1.0
    perclo: 5.0
    perchi: 95.0}
valnorm:
    impl: off
    rate: 0.01
    limit: 1e-8}
advnorm:
    impl: off
    rate: 0.01
    limit: 1e-8}
actent: 3e-4
slowreg: 1.0
slowtar: False
"""