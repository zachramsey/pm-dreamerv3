
import torch
from torchrl.data import LazyMemmapStorage, SliceSampler, TensorDictReplayBuffer
from tensordict import TensorDict
        

class ReplayBuffer:
    def __init__(self, cfg):
        self.cpu_device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_len = cfg["train_len"]
        self.obs_dim = cfg["obs_dim"]
        self.feat_dim = cfg["feat_dim"]
        self.act_dim = cfg["act_dim"]
        
        self.capacity = (cfg["replay_capacity"] // self.train_len) * self.train_len
        self.batch_ratio = cfg["batch_ratio"]
        self.batch_dim = cfg["batch_dim"]
        self.batch_len = cfg["batch_len"]

        self.episode = 0

        def collate_fn(data):
            return TensorDict({
                "x": torch.stack([d["x"] for d in data], dim=0),
                "a": torch.stack([d["a"] for d in data], dim=0),
                "r": torch.stack([d["r"] for d in data], dim=0)
            })

        # Create the replay buffer
        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(self.capacity, scratch_dir=cfg["scratch_dir"], device=self.cpu_device),
            collate_fn=collate_fn,
            # pin_memory=True,
            # prefetch=2,
            # batch_size=self.batch_len,
            dim_extend=0
        )

    def extend(self, observations, actions, rewards):
        # Ensure inputs have correct shapes
        assert observations.shape == (self.train_len, self.feat_dim)
        assert actions.shape == (self.train_len, self.act_dim)
        assert rewards.shape == (self.train_len,)

        # Create a TensorDict with the experience
        td = TensorDict({
            "episode": torch.full((self.train_len,), self.episode, dtype=torch.int64),
            "x": observations,
            "a": actions,
            "r": rewards
        }, batch_size=[self.train_len], device=self.cpu_device)

        # Extend buffer with the TensorDict
        self.buffer.extend(td)
        self.episode += 1

    def sample(self):
        batch_data = TensorDict({
            "x": torch.empty(self.batch_dim, self.batch_len, self.feat_dim, device=self.cpu_device),
            "a": torch.empty(self.batch_dim, self.batch_len, self.act_dim, device=self.cpu_device),
            "r": torch.empty(self.batch_dim, self.batch_len, device=self.cpu_device)
        })

        sample_random = len(self.buffer.storage) > self.train_len
        num_recents = int(self.batch_dim * (self.batch_ratio if sample_random else 1.0))
        num_randoms = self.batch_dim - num_recents

        recent_data = self.buffer.storage[-self.train_len:]
        random_data = self.buffer.storage[:-self.train_len]

        recent_idcs = torch.arange(start=0, end=len(recent_data), step=self.batch_len)
        random_idcs = torch.arange(start=0, end=len(random_data), step=self.batch_len)

        for i in range(num_recents):
            recent_idx = recent_idcs[torch.randint(0, len(recent_idcs), (1,)).item()]
            data = recent_data[recent_idx].unsqueeze(0)
            for j in range(recent_idx+1, recent_idx+self.batch_len):
                data = torch.cat((data, recent_data[j].unsqueeze(0)), dim=0)
            batch_data["x"][i] = data["x"]
            batch_data["a"][i] = data["a"]
            batch_data["r"][i] = data["r"]
        
        for i in range(num_randoms):
            random_idx = random_idcs[torch.randint(0, len(random_idcs), (1,)).item()]
            data = random_data[random_idx].unsqueeze(0)
            for j in range(random_idx+1, random_idx+self.batch_len):
                data = torch.cat((data, random_data[j].unsqueeze(0)), dim=0)
            batch_data["x"][i+num_recents] = data["x"]
            batch_data["a"][i+num_recents] = data["a"]
            batch_data["r"][i+num_recents] = data["r"]

        batch_data["x"] = batch_data["x"].transpose(0, 1)
        batch_data["a"] = batch_data["a"].transpose(0, 1)
        batch_data["r"] = batch_data["r"].transpose(0, 1)

        return batch_data.to(self.device)
