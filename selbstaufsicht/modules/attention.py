import torch
from torch import nn


class MultiHeadSelfAttention2d(nn.Module):
    def __init__(self, num_heads, dim_head, dropout=0.):
        super(MultiHeadSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.embed_dim = self.dim_head * self.num_heads

        self.in_projection = nn.Conv2d(self.embed_dim, 3*self.embed_dim, kernel_size=1)

    # TODO testing
    def forward(self, x, key_padding_mask=None, need_weights=True):
        """
        x: Tensor, batch first, channel first
        """
        B, D, S, L = x.size()
        assert D == self.embed_dim
        # TODO padding mask
        q, k, v = self.in_projection(x).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B * self.num_heads, self.dim_head, S * L)  # [B * H, DH, S * L]
        k = k.view(B * self.num_heads, self.dim_head, S * L)
        v = v.view(B * self.num_heads, self.dim_head, S * L)

        q = q * (self.dim_head ** -0.5)
        attn = torch.einsum('bci,bcj->bij', q, k)  # [B*H, S*L, S*L]
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bij,bdj->bdi', attn, v)  # [B*H, DH, S*L]
        out = out.view(B, D, S, L)  # [B, D, S, L]
        if need_weights:
            return out, attn
        return out


class AxialSelfAttention2d(nn.Module):
    def __init__(self):
        return


class TiedAxialSelfAttention2d(nn.Module):
    def __init__(self):
        return
