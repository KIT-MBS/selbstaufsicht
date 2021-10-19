import copy

import torch
from torch import nn

# TODO make c and d consistent
# TODO make S and H and L and W consistent
# TODO attention maps are formatted: [batch_size, num_heads, S, L], padding masks have to be adapted to that from [batch_size, S, L]


# TODO key padding masks
# NOTE dropout is applied analogously to pytorch attention: on the attention scores after applying softmax. don't know whether that makes sense. probably set to 0. anyways.
class MultiHeadSelfAttention2d(nn.Module):
    def __init__(self, num_heads, dim_head, dropout=0., device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.embed_dim = self.dim_head * self.num_heads

        self.in_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask=None, need_weights=True):
        """
        x: Tensor, batch first, channel first: [B, S, L, D]
        padding_mask: optional bool tensor: [B, S, L]
        """
        B, S, L, D = x.size()
        assert D == self.embed_dim

        # TODO test masking
        attn_mask = None
        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool
            assert padding_mask.size() == (B, S, L)
            padding_mask = padding_mask.view(B, 1, 1, 1, S, L).expand(-1, self.num_heads, -1, -1, -1, -1).reshape(B, self.num_heads, 1, 1, S, L)
            attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(padding_mask, float('-inf'))

        q, k, v = self.in_projection(x).chunk(3, dim=-1)  # [B, S, L, D]
        q = q.view(B, S * L, self.num_heads, self.dim_head)  # [B, S * L, H, DH]
        k = k.view(B, S * L, self.num_heads, self.dim_head)
        v = v.view(B, S * L, self.num_heads, self.dim_head)

        q = q * (self.dim_head ** -0.5)
        attn = torch.einsum('bihc,bjhc->bhij', q, k)  # [B, H, S*L, S*L]
        if attn_mask is not None:
            attn = attn.view(B, self.num_heads, S, L, S, L)
            attn += attn_mask
            attn = attn.view(B, self.num_heads, S * L, S * L)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bjhd->bihd', attn, v)  # [B, S*L, H, DH]
        # TODO optimize calls to contiguous
        out = out.contiguous()
        out = out.view(B, S, L, D)  # [B, S, L, D]
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

        self.in_row_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)
        self.in_col_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)

        self.norm1 = nn.LayerNorm(dim_head * num_heads, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(dim_head * num_heads, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask=None, need_weights=True, need_attn=True):
        """
        x: Tensor, batch first, channel first: [B, D, H, W]
        """
        B, S, L, D = x.size()
        assert D == self.embed_dim
        # TODO test masking
        attn_mask = None
        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool
            assert padding_mask.size() == (B, S, L)
            padding_mask = padding_mask.view(B, 1, S, 1, L).expand(-1, self.num_heads, -1, -1, -1).reshape(B, self.num_heads, S, 1, L)
            attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(padding_mask, float('-inf'))  # [B, S, L, H]

        q, k, v = self.in_row_projection(x).chunk(3, dim=-1)  # [B, S, L, D]
        q = q.view(B, S, L, self.num_heads, self.dim_head)  # [B, S, L, H, DH]
        k = k.view(B, S, L, self.num_heads, self.dim_head)
        v = v.view(B, S, L, self.num_heads, self.dim_head)

        # NOTE row attn
        row_attn = torch.einsum('bhcsi, bhcsj->bhsij', q, k)  # [B, H, S, L, L]
        row_attn = torch.einsum('bsihc, bsjhc->bhsij', q, k)  # [B, H, S, L, L]
        if attn_mask is not None:
            row_attn_mask = attn_mask.expand(-1, -1, -1, L, -1)
            print("row_attn", row_attn.shape)
            print("row_attn_mask", row_attn_mask.shape)
            row_attn += row_attn_mask
        row_attn = row_attn.softmax(dim=-1)
        row_attn = self.dropout1(row_attn)
        row_out = torch.einsum('bhsij, bsjhd->bsihd', row_attn, v)

        # NOTE: view not possible here, need reshape
        row_out = row_out.reshape(B, S, L, D)
        out = x + row_out
        out = self.norm1(out)

        q, k, v = self.in_col_projection(out).chunk(3, dim=-1)  # [B, S, L, D]
        q = q.view(B, S, L, self.num_heads, self.dim_head)  # [B, S, L, H, DH]
        k = k.view(B, S, L, self.num_heads, self.dim_head)
        v = v.view(B, S, L, self.num_heads, self.dim_head)

        # NOTE col attn
        col_attn = torch.einsum('bilhc, bjlhc->bhijl', q, k)  # [B, H, S, S, L]
        if attn_mask is not None:
            col_attn_mask = attn_mask.expand(-1, -1, -1, S, -1)
            print("col_attn", col_attn.shape)
            print("col_attn_mask", col_attn_mask.shape)
            col_attn += col_attn_mask
        col_attn = col_attn.softmax(dim=-2)
        col_attn = self.dropout2(col_attn)
        col_out = torch.einsum('bhijl, bjlhd->bilhd', col_attn, v)

        # NOTE: view not possible here, need reshape
        col_out = col_out.reshape(B, S, L, D)

        out = out + col_out
        out = self.norm2(out)

        if need_attn:
            return out, (row_attn, col_attn)
        return out


# NOTE difference to original tied axial attention: Row attention done first, to have something akin to a learned positional embedding along the evolutionary axis
class TiedAxialSelfAttention2d(nn.Module):
    def __init__(self, num_heads, dim_head, dropout=0., layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TiedAxialSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.embed_dim = self.dim_head * self.num_heads

        self.in_row_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)
        self.in_col_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)

        self.norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask=None, need_attn=True):
        """
        x: Tensor, batch first, channel first: [B, D, S, L]
        """
        B, S, L, D = x.size()
        assert D == self.embed_dim
        # TODO test padding mask
        attn_mask = None
        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool
            assert padding_mask.size() == (B, S, L)
            padding_mask = padding_mask.view(B, 1, S, L).expand(-1, self.num_heads, -1, -1).reshape(B, self.num_heads, S, L)
            attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(padding_mask, float('-inf'))  # [B, H, S, L]

        q, k, v = self.in_row_projection(x).chunk(3, dim=-1)  # [B, S, L, D]
        q = q.view(B, S, L, self.num_heads, self.dim_head)  # [B, S, L, H, DH]
        k = k.view(B, S, L, self.num_heads, self.dim_head)
        v = v.view(B, S, L, self.num_heads, self.dim_head)

        # NOTE row attn
        row_attn = torch.einsum('bsihc, bsjhc->bhij', q, k)  # [B, H, L, L]
        if attn_mask is not None:
            row_attn += attn_mask.sum(-2).unsqueeze(-1)
        row_attn = row_attn.unsqueeze(2)  # [B, H, 1, L ,L]
        row_attn = row_attn.softmax(dim=-1)  # [B, H, 1, L, L]
        row_attn = self.dropout1(row_attn)
        row_out = torch.einsum('bhsij, bsjhd->bsihd', row_attn, v)
        row_out = row_out.reshape(B, S, L, D)

        out = x + row_out
        out = self.norm1(out)

        q, k, v = self.in_col_projection(out).chunk(3, dim=-1)  # [B, S, L, D]
        q = q.view(B, S, L, self.num_heads, self.dim_head)  # [B, H, DH, S, L]
        k = k.view(B, S, L, self.num_heads, self.dim_head)
        v = v.view(B, S, L, self.num_heads, self.dim_head)

        # NOTE col attn
        col_attn = torch.einsum('bilhc, bjlhc->bhijl', q, k)  # [B, H, S, S, L]
        if attn_mask is not None:
            row_attn += attn_mask.unsqueeze(-3)

        col_attn = col_attn.softmax(dim=-2)
        col_attn = self.dropout2(col_attn)
        col_out = torch.einsum('bhijl, bjlhd->bilhd', col_attn, v)  # [B, S, L, H, DH]
        col_out = col_out.reshape(B, S, L, D)

        out = out + col_out
        out = self.norm2(out)

        if need_attn:
            return out, (row_attn, col_attn)
        return out


# TODO not sure about the name, self attention, transform oneself... mutate... morph meh
class Transmorpher2d(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super(Transmorpher2d, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    # TODO need attn is a bit clunkily done
    def forward(self, x, padding_mask=None, need_attn=False):
        out = x
        if need_attn:
            attns = []
            for layer in self.layers:
                out, a = layer(out, padding_mask=padding_mask, need_attn=need_attn)
                attns.append(a)
            if self.norm is not None:
                out = self.norm(out)
            return out, attns

        for layer in self.layers:
            out = layer(out, padding_mask=padding_mask, need_attn=need_attn)
        if self.norm is not None:
            out = self.norm(out)

        return out


class TransmorpherLayer2d(nn.Module):
    def __init__(self, dim_head, num_heads, dim_ff, dropout=0.1, attention='tied', activation='relu', layer_norm_eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        dim_model = dim_head * num_heads
        super(TransmorpherLayer2d, self).__init__()
        self.attn = _get_attention_function(attention)(dim_head, num_heads, dropout=dropout, **factory_kwargs)

        self.lin1 = nn.Linear(dim_model, dim_ff, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_ff, dim_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_function(activation)()

    def forward(self, x, padding_mask=None, need_attn=False):
        out = self.attn(x, padding_mask=padding_mask, need_attn=need_attn)
        if need_attn:
            out, attn = out
        # TODO what's the last layer of an attention block? should it be a nonlinearity
        x = x + self.dropout1(out)
        x = self.norm1(out)
        out = self.lin2(self.dropout(self.activation(self.lin1(x))))
        x = x + self.dropout2(out)
        x = self.norm2(x)
        if need_attn:
            return x, attn
        return x


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
        return nn.ReLU
    elif activation == 'gelu':
        return nn.GELU
    raise RuntimeError("Expected relu or gelu not {}".format(activation))
