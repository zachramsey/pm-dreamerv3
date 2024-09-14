
import torch
import torch.nn as nn

#----------------------------------------------------------------------------------------------------

class ModularMLP(nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        super(ModularMLP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_dim = cfg["hidden_dim"]
        act = getattr(nn, cfg["activation"])()
        normalize = cfg["normalize"]
        num_layers = cfg["num_hidden_layers"]
        assert num_layers > 0, "ModularMLP: num_hidden_layers must be greater than 0"

        self.model = nn.Sequential()

        # Input layer
        self.model.add_module(f'linear_input', nn.Linear(input_dim, hidden_dim, device=self.device))
        self.model.add_module(f'activation_input', act)
        if normalize:
            self.model.add_module(f'norm_input', nn.RMSNorm(hidden_dim, device=self.device))

        # Additional hidden layers
        for i in range(1, num_layers):
            self.model.add_module(f'linear_{i}', nn.Linear(hidden_dim, hidden_dim, device=self.device))
            self.model.add_module(f'activation_{i}', act)
            if normalize:
                self.model.add_module(f'norm_{i}', nn.RMSNorm(hidden_dim, device=self.device))

        # Output layer
        self.model.add_module(f'linear_output', nn.Linear(hidden_dim, output_dim, device=self.device))

        if cfg["init_zeros"]:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

    def forward(self, x):
        return self.model(x)
    
#----------------------------------------------------------------------------------------------------

class AttnBlock(nn.Module):
    def __init__(self, cfg):
        super(AttnBlock, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emb_dim = cfg["feat_dim"]
        self.num_heads = cfg["embedder"]["num_heads"]
        self.dropout = cfg["embedder"]["dropout"]

        self.attn = nn.MultiheadAttention(self.emb_dim, self.num_heads, self.dropout, batch_first=True, device=self.device)
        self.norm1 = nn.LayerNorm(self.emb_dim, device=self.device)
        self.mlp = ModularMLP(self.emb_dim, self.emb_dim, cfg["embedder"])
        self.norm2 = nn.LayerNorm(self.emb_dim, device=self.device)

    def forward(self, x):
        attn, _ = self.attn(x, x, x)
        x = self.norm1(x + attn)
        ffn = self.mlp(x)
        x = self.norm2(x + ffn)
        return x
    
class ModularAttn(nn.Module):
    def __init__(self, cfg):
        super(ModularAttn, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_layers = cfg["embedder"]["num_attn_layers"]
        assert num_layers > 0, "ModularAttn: num_hidden_layers must be greater than 0"

        self.model = nn.Sequential()

        # Input layer
        self.model.add_module(f'attn_input', AttnBlock(cfg))

        # Additional hidden layers
        for i in range(1, num_layers):
            self.model.add_module(f'attn_{i}', AttnBlock(cfg))

        # Output layer
        # self.model.add_module(f'linear_output', nn.Linear(cfg["feat_dim"], cfg["feat_dim"], device=self.device))

    def forward(self, x):
        return self.model(x)
    
#----------------------------------------------------------------------------------------------------

# class BlockDiagonalGRUCell(nn.Module):
#     def __init__(self, input_size, hidden_dim, num_blocks):
#         super(BlockDiagonalGRUCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_dim = hidden_dim
#         self.num_blocks = num_blocks
        
#         # Ensure hidden_dim is divisible by num_blocks
#         assert hidden_dim % num_blocks == 0, "hidden_dim must be divisible by num_blocks"
#         self.block_size = hidden_dim // num_blocks
        
#         # Create block-diagonal weight matrices
#         self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, input_size))
#         self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
#         self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
#         self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_dim))
        
#         for weight in [self.weight_ih, self.weight_hh]:
#             nn.init.orthogonal_(weight)
#         nn.init.zeros_(self.bias_ih)
#         nn.init.zeros_(self.bias_hh)

#     def forward(self, input, hx=None):
#         if hx is None:
#             hx = torch.zeros(input.size(0), self.hidden_dim, dtype=input.dtype, device=input.device)
        
#         # Compute the input-to-hidden contribution
#         ih = torch.matmul(input, self.weight_ih.t()) + self.bias_ih
        
#         # Compute the hidden-to-hidden contribution
#         hh = torch.zeros_like(ih)
#         for i in range(self.num_blocks):
#             start = i * self.block_size
#             end = (i + 1) * self.block_size
#             hh[:, start*3:end*3] = torch.matmul(
#                 hx[:, start:end],
#                 self.weight_hh[start*3:end*3, start:end].t()
#             ) + self.bias_hh[start*3:end*3]
        
#         # Gates
#         gates = ih + hh
#         r, z, n = gates.chunk(3, 1)
        
#         # Apply activations
#         r = torch.sigmoid(r)
#         z = torch.sigmoid(z)
#         n = torch.tanh(n)
        
#         # Compute new hidden state
#         hy = (1 - z) * n + z * hx
        
#         return hy
    

# class BlockDiagonalGRUCell(nn.Module):
#     def __init__(self, input_size, hidden_dim, num_blocks):
#         super(BlockDiagonalGRUCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_dim = hidden_dim
#         self.num_blocks = num_blocks
        
#         # Ensure that input_size and hidden_dim are divisible by num_blocks
#         assert input_size % num_blocks == 0, "input_size must be divisible by num_blocks"
#         assert hidden_dim % num_blocks == 0, "hidden_dim must be divisible by num_blocks"
        
#         self.block_input_size = input_size // num_blocks
#         self.block_hidden_size = hidden_dim // num_blocks
        
#         # Define GRU gates (reset, update) and candidate hidden state for each block
#         self.W_ir = nn.ModuleList([nn.Linear(self.block_input_size, self.block_hidden_size) for _ in range(num_blocks)])
#         self.W_iz = nn.ModuleList([nn.Linear(self.block_input_size, self.block_hidden_size) for _ in range(num_blocks)])
#         self.W_in = nn.ModuleList([nn.Linear(self.block_input_size, self.block_hidden_size) for _ in range(num_blocks)])
        
#         self.W_hr = nn.ModuleList([nn.Linear(self.block_hidden_size, self.block_hidden_size, bias=False) for _ in range(num_blocks)])
#         self.W_hz = nn.ModuleList([nn.Linear(self.block_hidden_size, self.block_hidden_size, bias=False) for _ in range(num_blocks)])
#         self.W_hn = nn.ModuleList([nn.Linear(self.block_hidden_size, self.block_hidden_size, bias=False) for _ in range(num_blocks)])

#     def forward(self, x, h):
#         # Split input and hidden states into blocks
#         x_blocks = x.split(self.block_input_size, dim=1)
#         h_blocks = h.split(self.block_hidden_size, dim=1)
        
#         new_h_blocks = []
        
#         # Compute GRU update for each block independently
#         for i in range(self.num_blocks):
#             x_b = x_blocks[i]
#             h_b = h_blocks[i]
            
#             # Compute reset and update gates
#             r = torch.sigmoid(self.W_ir[i](x_b) + self.W_hr[i](h_b))
#             z = torch.sigmoid(self.W_iz[i](x_b) + self.W_hz[i](h_b))
            
#             # Compute candidate hidden state
#             n = torch.tanh(self.W_in[i](x_b) + r * self.W_hn[i](h_b))
            
#             # Compute new hidden state
#             new_h_b = (1 - z) * n + z * h_b
#             new_h_blocks.append(new_h_b)
        
#         # Concatenate updated hidden states from all blocks
#         new_h = torch.cat(new_h_blocks, dim=1)
#         return new_h
    

# class BlockDiagonalGRUCell(nn.Module):
#     def __init__(self, input_size, hidden_dim, num_blocks):
#         super(BlockDiagonalGRUCell, self).__init__()

#         self.input_size = input_size
#         self.hidden_dim = hidden_dim
#         self.num_blocks = num_blocks
#         self.block_size = hidden_dim // num_blocks

#         # Check if the hidden size is evenly divisible by the number of blocks
#         assert self.hidden_dim % self.num_blocks == 0, "Hidden size must be divisible by the number of blocks"

#         # Create update, reset, and new gate matrices
#         self.update_gate = nn.Linear(input_size + self.hidden_dim, self.hidden_dim)
#         self.reset_gate = nn.Linear(input_size + self.hidden_dim, self.hidden_dim)
#         self.new_gate = nn.Linear(input_size + self.hidden_dim, self.hidden_dim)

#     def forward(self, input, hidden):
#         # Concatenate input and hidden state
#         combined = torch.cat([input, hidden], dim=1)

#         # Compute update, reset, and new gates
#         update_gate = torch.sigmoid(self.update_gate(combined))
#         reset_gate = torch.sigmoid(self.reset_gate(combined))
#         new_gate = torch.tanh(self.new_gate(torch.cat([input, reset_gate * hidden], dim=1)))

#         # Compute new hidden state
#         hidden = update_gate * hidden + (1 - update_gate) * new_gate

#         # Reshape hidden state to block-diagonal form
#         hidden = hidden.view(hidden.size(0), self.num_blocks, self.block_size)
#         hidden = torch.diagonal(hidden, dim1=1, dim2=2).view(hidden.size(0), -1)

#         return hidden
    
# class BlockLinear(nn.Module):
#     def __init__(self, units, groups, act='none', outscale=1.0,
#                  winit='normal', binit=False, fan='in', dtype=torch.float32, block_fans=False, block_norm=False):
#         super(BlockLinear, self).__init__()
#         self.units = (units,) if isinstance(units, int) else tuple(units)
#         assert groups <= torch.prod(self.units), (groups, self.units)
#         self.groups = groups
#         self.act = act
#         self.outscale = outscale
#         self.dtype = dtype
#         self.block_fans = block_fans
#         self.block_norm = block_norm

#         in_feats = self.groups * torch.prod(self.units)
#         out_feats = torch.prod(self.units)

#         group_in_feats = in_feats // self.groups
#         group_out_feats = out_feats // self.groups
#         std = self.outscale / torch.sqrt(in_feats)

#         if self.block_norm:
#             self.norm_layers = nn.ModuleList([self._create_norm_layer() for _ in range(self.groups)])
#         else:
#             self.norm_layers = self._create_norm_layer()

#         self.weight = nn.Parameter(torch.randn(self.groups, group_in_feats, group_out_feats) * std)
#         self.bias = nn.Parameter(torch.zeros(out_feats))

#     def _create_norm_layer(self):
#         return nn.BatchNorm1d(torch.prod(self.units))

#     def forward(self, x):
#         x = self._layer(x)
#         if self.block_norm:
#             x = torch.cat([
#                 norm_layer(x[:, i * x.size(1) // self.groups:(i + 1) * x.size(1) // self.groups])
#                 for i, norm_layer in enumerate(self.norm_layers)
#             ], dim=1)
#         else:
#             x = self.norm_layers(x)
#         x = self._apply_activation(x)
#         return x

#     def _layer(self, x):
#         indim, outdim = x.shape[-1], torch.prod(self.units)
#         assert indim % self.groups == outdim % self.groups == 0, (indim, outdim, self.groups, self.units)
#         x = x.view(-1, self.groups, indim // self.groups)
#         x = torch.einsum('...ki,kio->...ko', x, self.weight)
#         x = x.view(-1, outdim)
#         x += self.bias
#         if len(self.units) > 1:
#             x = x.view(x.shape[0], *self.units)
#         return x


# class BlockDiagonalGRUCell(nn.Module):
#     def __init__(self, cfg):
#         super(BlockDiagonalGRUCell, self).__init__()

#         self.recur_dim = cfg["recurrent_dim"]
#         self.latent_dim = cfg["stochastic_dim"] * cfg["discrete_dim"]
#         self.action_dim = cfg["obs_dim"]

#         self.hidden_dim = cfg["hidden_dim"]
#         self.num_blocks = cfg["num_blocks"]
#         self.block_size = self.hidden_dim // self.num_blocks

#         assert self.hidden_dim % self.num_blocks == 0, "Hidden size must be divisible by the number of blocks"

#         self.flat2group = lambda x: x.view(-1, self.num_blocks, self.block_size)
#         self.group2flat = lambda x: x.view(-1, self.hidden_dim)

#         self.h_in = nn.Linear(self.recur_dim, self.hidden_dim)
#         self.z_in = nn.Linear(self.latent_dim, self.hidden_dim)
#         self.a_in = nn.Linear(self.action_dim, self.hidden_dim)

#         self.dyn_layers = nn.ModuleList([
#             BlockLinear(

#     def forward(self, h, z, a):
#         x0 = self.h_in(h)
#         x1 = self.z_in(z)
#         x2 = self.a_in(a)

#         x = torch.cat([x0, x1, x2], dim=1).unsqueeze(-2).repeat(1, 1, self.num_blocks, 1)
#         x = self.group2flat(torch.cat([self.flat2group(z, self.num_blocks), x], dim=-1))

