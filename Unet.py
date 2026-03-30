import torch
import torch.nn as nn
from einops import rearrange
from .utils import PositionalEncoding


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
        super().__init__()

        self.dim, self.dim_out = dim, dim_out

        dim_out = dim if dim_out is None else dim_out
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), padding=1)
        self.block1 = nn.Sequential(self.norm1, self.activation1, self.conv1)

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim is not None else None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), padding=1)
        self.block2 = nn.Sequential(self.norm2, self.activation2, self.dropout, self.conv2)

        self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        hidden = self.block1(x)
        if time_emb is not None:
            # add in timestep embedding
            hidden = hidden + self.mlp(time_emb)[..., None, None]  # (B, dim_out, 1, 1)
        hidden = self.block2(hidden)
        return hidden + self.residual_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, groups=32, heads=4):
        super().__init__()
        self.dim, self.dim_out = dim, dim
        self.heads = heads
        self.head_dim = dim // heads

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1))
        self.to_out = nn.Conv2d(dim, dim, kernel_size=(1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (heads d) h w -> b heads d (h w)',
                                           heads=self.heads), qkv)

        # Kernel trick: use ELU+1 as feature map φ(x)
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        # Linear attention: Q(KᵀV) instead of softmax(QKᵀ)V
        k_sum = k.sum(dim=-1, keepdim=True)          # normalization term
        kv = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # (KᵀV)
        out = torch.einsum('b h d e, b h d n -> b h e n', kv, q)  # Q(KᵀV)

        # Normalize
        denom = torch.einsum('b h d n, b h d s -> b h n s', q, k_sum).squeeze(-1)
        out = out / (denom.unsqueeze(2) + 1e-6)

        out = rearrange(out, 'b heads d (h w) -> b (heads d) h w', h=h, w=w)
        return self.to_out(out) + x


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, groups=32):
        super().__init__()
        self.dim, self.dim_out = dim, dim

        self.scale = dim ** (-0.5)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.norm_context = nn.LayerNorm(context_dim)

        # Q comes from image features, K and V come from context
        self.to_q  = nn.Conv2d(dim, dim, kernel_size=(1, 1))
        self.to_kv = nn.Linear(context_dim, dim * 2)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=(1, 1))

    def forward(self, x, context):
        # x:       (B, C, H, W)  — image features
        # context: (B, seq_len, context_dim)  — e.g. text tokens
        b, c, h, w = x.shape

        q = self.to_q(self.norm(x))
        q = rearrange(q, 'b c h w -> b (h w) c')

        context = self.norm_context(context)
        k, v = self.to_kv(context).chunk(2, dim=-1)  # both: (B, seq_len, dim)

        similarity = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale
        attn = torch.softmax(similarity, dim=-1)
        out = torch.einsum('b i j, b j c -> b i c', attn, v)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return self.to_out(out) + x



class ResnetAttentionBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32, context_dim=None):
        super().__init__()

        self.dim, self.dim_out = dim, dim_out

        self.resnet = ResnetBlock(dim, dim_out, time_emb_dim, dropout, groups)
        self.attention = LinearAttention(dim_out, groups)
        self.cross_attn = CrossAttention(dim_out, context_dim, groups) \
                          if context_dim is not None else None

    def forward(self, x, time_emb=None, context=None):
        x = self.resnet(x, time_emb)
        x= self.attention(x)
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context)
        
        return x
        


class downSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.dim, self.dim_out = dim_in, dim_in

        self.downsameple = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        return self.downsameple(x)


class upSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.dim, self.dim_out = dim_in, dim_in

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        return self.upsample(x)


class Unet(nn.Module):
    def __init__(self, dim, image_size, dim_multiply=(1, 2, 4, 8), channel=1, num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0, device='cuda', groups=32, context_dim=None):

        super().__init__()
        assert dim % groups == 0, 'parameter [groups] must be divisible by parameter [dim]'

        # Attributes
        self.dim = dim
        self.channel = channel
        self.time_emb_dim = 4 * self.dim
        self.num_resolutions = len(dim_multiply)
        self.device = device
        self.resolution = [int(image_size / (2 ** i)) for i in range(self.num_resolutions)]
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]
        self.num_res_blocks = num_res_blocks

        # Time embedding
        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.SiLU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Layer definition
        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])
        concat_dim = list()

        # Downward Path layer definition
        self.init_conv = nn.Conv2d(channel, self.dim, kernel_size=(3, 3), padding=1)
        concat_dim.append(self.dim)

        for level in range(self.num_resolutions):
            d_in, d_out = self.hidden_dims[level], self.hidden_dims[level + 1]
            for block in range(num_res_blocks):
                d_in_ = d_in if block == 0 else d_out
                if self.resolution[level] in attn_resolutions:
                    self.down_path.append(ResnetAttentionBlock(d_in_, d_out, self.time_emb_dim, dropout, groups, context_dim=context_dim))
                else:
                    self.down_path.append(ResnetBlock(d_in_, d_out, self.time_emb_dim, dropout, groups))
                concat_dim.append(d_out)
            if level != self.num_resolutions - 1:
                self.down_path.append(downSample(d_out))
                concat_dim.append(d_out)

        # Middle layer definition
        mid_dim = self.hidden_dims[-1]
        self.middle_resnet_attention = ResnetAttentionBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups, context_dim=context_dim)
        self.middle_resnet = ResnetBlock(mid_dim, mid_dim, self.time_emb_dim, dropout, groups)

        # Upward Path layer definition
        for level in reversed(range(self.num_resolutions)):
            d_out = self.hidden_dims[level + 1]
            for block in range(num_res_blocks + 1):
                d_in = self.hidden_dims[level + 2] if block == 0 and level != self.num_resolutions - 1 else d_out
                d_in = d_in + concat_dim.pop()
                if self.resolution[level] in attn_resolutions:
                    self.up_path.append(ResnetAttentionBlock(d_in, d_out, self.time_emb_dim, dropout, groups, context_dim=context_dim))
                else:
                    self.up_path.append(ResnetBlock(d_in, d_out, self.time_emb_dim, dropout, groups))
            if level != 0:
                self.up_path.append(upSample(d_out))

        assert not concat_dim, 'Error in concatenation between downward path and upward path.'

        # Output layer
        final_ch = self.hidden_dims[1]
        self.final_norm = nn.GroupNorm(groups, final_ch)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(final_ch, channel, kernel_size=(3, 3), padding=1)

    def forward(self, x, time, context=None):

        t = self.time_mlp(time)
        # Downward
        concat = list()
        x = self.init_conv(x)
        concat.append(x)
        for layer in self.down_path:
            if isinstance(layer, downSample):
                x = layer(x)
            elif isinstance(layer, ResnetAttentionBlock):
                x = layer(x, t, context)
            else:  # ResnetBlock
                x = layer(x, t)
            concat.append(x)

        # Middle
        x = self.middle_resnet_attention(x, t, context)
        x = self.middle_resnet(x, t)

        # Upward
        for layer in self.up_path:
            if isinstance(layer, upSample):
                x = layer(x)
            else:
                x = torch.cat((x, concat.pop()), dim=1)
                if isinstance(layer, ResnetAttentionBlock):
                    x = layer(x, t, context)
                else:  # ResnetBlock
                    x = layer(x, t)

        # Final
        x = self.final_activation(self.final_norm(x))
        return self.final_conv(x)

    def print_model_structure(self):
        for i in self.down_path:
            if i.__class__.__name__ == 'downSample':
                print('-' * 20)
            if i.__class__.__name__ == "Conv2d":

                print(i.__class__.__name__)
            else:
                print(i.__class__.__name__, i.dim, i.dim_out)
        print('\n')
        print('=' * 20)
        print('\n')
        for i in self.up_path:
            if i.__class__.__name__ == 'upSample':
                print('-' * 20)
            if i.__class__.__name__ == "Conv2d":
                print(i.__class__.__name__)
            else:
                print(i.__class__.__name__, i.dim, i.dim_out)
