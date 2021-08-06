import torch
from torch import nn
from .layernorm import AxialLayerNorm

# TODO all factory kwargs
# TODO key padding masks
# NOTE dropout is applied analogously to pytorch attention: on the attention scores after applying softmax. don't know whether that makes sense. probably set to 0. anyways.
class MultiHeadSelfAttention2d(nn.Module):
    def __init__(self, num_heads, dim_head, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.embed_dim = self.dim_head * self.num_heads

        self.in_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)
        self.norm = AxialLayerNorm(1, dim_head * num_heads, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, key_padding_mask=None, need_weights=True):
        """
        x: Tensor, batch first, channel first: [B, D, H, W]
        key_padding_mask: optional bool tensor: [B, H, W]
        """
        B, D, S, L = x.size()
        assert D == self.embed_dim

        # TODO test masking
        attn_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size() == (B, S, L)
            key_padding_mask = key_padding_mask.view(B, 1, 1, S, L).expand(-1, self.num_heads, -1, -1, -1).reshape(B, self.num_heads, 1, S, L)
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(attn_mask, float('-inf'))

        q, k, v = self.in_projection(x).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B, self.num_heads, self.dim_head, S * L)  # [B, H, DH, S * L]
        k = k.view(B, self.num_heads, self.dim_head, S * L)
        v = v.view(B, self.num_heads, self.dim_head, S * L)

        q = q * (self.dim_head ** -0.5)
        attn = torch.einsum('bhci,bhcj->bhij', q, k)  # [B, H, S*L, S*L]
        if attn_mask is not None:
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)  # [B, H, DH, S*L]
        # TODO optimize calls to contiguous
        out = out.contiguous()
        out = out.view(B, D, S, L)  # [B, D, S, L]
        if need_weights:
            return out, attn
        return out


class AxialSelfAttention2d(nn.Module):
    # TODO optimize einsum strings using opt_einsum package
    def __init__(self, num_heads, dim_head, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AxialSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.embed_dim = self.dim_head * self.num_heads

        self.in_row_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)
        self.in_col_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)

        self.norm1 = AxialLayerNorm(1, dim_head * num_heads, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = AxialLayerNorm(1, dim_head * num_heads, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

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
        row_attn = self.dropout1(row_attn)
        row_out = torch.einsum('bhsij, bhdsj->bhdsi', row_attn, v)

        # TODO: view not possible here, need reshape
        #row_out = row_out.view(B, D, S, L)
        row_out = row_out.reshape(B, D, S, L)
        out = x + row_out
        out = self.norm1(out)

        q, k, v = self.in_col_projection(out).chunk(3, dim=1)  # [B, D, S, L]
        q = q.view(B, self.num_heads, self.dim_head, S, L)  # [B, H, DH, S, L]
        k = k.view(B, self.num_heads, self.dim_head, S, L)
        v = v.view(B, self.num_heads, self.dim_head, S, L)

        # NOTE col attn
        col_attn = torch.einsum('bhcil, bhcjl->bhijl', q, k)
        col_attn = col_attn.softmax(dim=-2)
        col_attn = self.dropout2(col_attn)
        col_out = torch.einsum('bhijl, bhdjl->bhdil', col_attn, v)

        # TODO: view not possible here, need reshape
        #col_out = col_out.view(B, D, S, L)
        col_out = col_out.reshape(B, D, S, L)

        out = out + col_out
        out = self.norm2(out)

        return out


# NOTE difference to original tied axial attention: Row attention done first, to have something akin to a learned positional embedding along the evolutionary axis
class TiedAxialSelfAttention2d(nn.Module):
    def __init__(self, num_heads, dim_head, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TiedAxialSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.embed_dim = self.dim_head * self.num_heads

        self.in_row_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)
        self.in_col_projection = nn.Conv2d(self.embed_dim, 3 * self.embed_dim, kernel_size=1, **factory_kwargs)

        self.norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

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
        row_attn = self.dropout1(row_attn)
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
        col_attn = self.dropout(col_attn)
        col_out = torch.einsum('bhijl, bhdjl->bhdil', col_attn, v)
        col_out = col_out.view(B, D, S, L)

        out = out + col_out
        out = self.norm2(out)

        return out


# TODO not sure about the name, self attention, transform oneself... mutate... morph meh
class Transmorpher(nn.Module):
    def __init__(self, attention_flavor, d, num_heads, d_ff, dropout=0., device=None, dtype=None):
        return

    def forward(self, x, key_padding_mask, need_attn=False):
        return


class TransmorpherLayer(nn.Module):
    def __init__(self, dim_head, num_heads, dim_ff, dropout=0.1, attention='', activation='relu', layer_norm_eps=1e-5, device=None, dtype=None):
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransmorpherLayer, self).__init__()
        return

    def forward(self, x, key_padding_mask, need_attn=False):
        return


def _get_attention_function(attention):
    if attention == 'full':
        return MultiHeadSelfAttention2d
    elif attention == 'axial':
        return AxialSelfAttention2d
    elif attention == 'tied':
        return TiedAxialSelfAttention2d
    raise RuntimeError("Expected full, axial, or tied not {}".format(attention))


def _get_activation_function(activation):
    if activation == 'relu':
        return nn.Relu
    elif activation == 'gelu':
        return nn.GELU
    raise RuntimeError("Expected relu or gelu not {}".format(activation))
