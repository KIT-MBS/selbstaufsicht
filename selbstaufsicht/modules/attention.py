import torch
from torch import nn


# TODO all factory kwargs
# TODO key padding masks
# TODO dropout
class MultiHeadSelfAttention2d(nn.Module):
    def __init__(self, num_heads, dim_head, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        if dropout > 0.:
            raise NotImplementedError()
        self.embed_dim = self.dim_head * self.num_heads

        self.in_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)
        self.norm = nn.LaynerNorm(dim_head * num_heads, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=True):
        """
        x: Tensor, batch first, channel first: [B, D, H, W]
        """
        B, D, S, L = x.size()
        assert D == self.embed_dim
        # TODO padding mask
        q, k, v = self.in_projection(x).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B, self.num_heads, self.dim_head, S * L)  # [B, H, DH, S * L]
        k = k.view(B, self.num_heads, self.dim_head, S * L)
        v = v.view(B, self.num_heads, self.dim_head, S * L)

        q = q * (self.dim_head ** -0.5)
        attn = torch.einsum('bhci,bhcj->bhij', q, k)  # [B, H, S*L, S*L]
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)  # [B, H, DH, S*L]
        # TODO optimize calls to contiguous
        out = out.contiguous()
        out = out.view(B, D, S, L)  # [B, D, S, L]
        if need_weights:
            return out, attn
        return out


class AxialSelfAttention2d(nn.Module):
    # TODO dropout
    # TODO optimize einsum strings using opt_einsum package
    def __init__(self, num_heads, dim_head, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AxialSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        if dropout > 0.:
            raise NotImplementedError()
        self.embed_dim = self.dim_head * self.num_heads
        self.in_row_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1)
        self.in_col_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=True):
        """
        x: Tensor, batch first, channel first: [B, D, H, W]
        """
        B, D, S, L = x.size()
        assert D == self.embed_dim
        # TODO padding mask

        q, k, v = self.in_row_projection(x).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B, self.num_heads, self.dim_head, S, L)  # [B, H, DH, S, L]
        k = k.view(B, self.num_heads, self.dim_head, S, L)
        v = v.view(B, self.num_heads, self.dim_head, S, L)

        # NOTE row attn
        row_attn = torch.einsum('bhcsi, bhcsj->bhsij', q, k)
        row_attn = row_attn.softmax(dim=-1)
        row_out = torch.einsum('bhsij, bhdsj->bhdsi', row_attn, v)
        row_out = row_out.view(B, D, S, L)
        out = x + row_out
        out = self.norm1(out)

        q, k, v = self.in_col_projection(out).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B, self.num_heads, self.dim_head, S, L)  # [B, H, DH, S, L]
        k = k.view(B, self.num_heads, self.dim_head, S, L)
        v = v.view(B, self.num_heads, self.dim_head, S, L)

        # NOTE col attn
        col_attn = torch.einsum('bhcil, bhcjl->bhijl', q, k)
        col_attn = col_attn.softmax(dim=-2)
        col_out = torch.einsum('bhijl, bhdjl->bhdil', col_attn, v)
        col_out = col_out.view(B, D, S, L)

        out = out + col_out
        out = self.norm2(out)

        return out


class TiedAxialSelfAttention2d(nn.Module):
    def __init__(self, num_heads, dim_head, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TiedAxialSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        if dropout > 0.:
            raise NotImplementedError()
        self.embed_dim = self.dim_head * self.num_heads
        self.in_row_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)
        self.in_col_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=True):
        """
        x: Tensor, batch first, channel first: [B, D, H, W]
        """
        B, D, S, L = x.size()
        assert D == self.embed_dim
        # TODO padding mask
        q, k, v = self.in_row_projection(x).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B, self.num_heads, self.dim_head, S, L)  # [B, H, DH, S, L]
        k = k.view(B, self.num_heads, self.dim_head, S, L)
        v = v.view(B, self.num_heads, self.dim_head, S, L)

        # NOTE row attn
        row_attn = torch.einsum('bhcsi, bhcsj->bhij', q, k)
        row_attn = row_attn.unsqueeze(2)
        row_attn = row_attn.softmax(dim=-1)  # [B, H, 1, L, L]
        row_out = torch.einsum('bhsij, bhdsj->bhdsi', row_attn, v)
        row_out = row_out.view(B, D, S, L)

        out = x + row_out
        out = self.norm1(out)

        q, k, v = self.in_col_projection(out).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B, self.num_heads, self.dim_head, S, L)  # [B, H, DH, S, L]
        k = k.view(B, self.num_heads, self.dim_head, S, L)
        v = v.view(B, self.num_heads, self.dim_head, S, L)

        # NOTE col attn
        col_attn = torch.einsum('bhcil, bhcjl->bhijl', q, k)
        col_attn = col_attn.softmax(dim=-2)
        col_out = torch.einsum('bhijl, bhdjl->bhdil', col_attn, v)
        col_out = col_out.view(B, D, S, L)

        out = out + col_out
        out = self.norm2(out)

        return out


class TransformerEncoderLayer():
    def __init__(self, attention_flavor, d, num_heads, d_ff, dropout=0., device=None, dtype=None):
        return

    def forward(self, x, key_padding_mask, need_attn=False):
        return
