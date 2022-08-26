import copy
from typing import List, Tuple, Type, Union

import math
import torch
from torch import nn


# NOTE dropout is applied analogously to pytorch attention: on the attention scores after applying softmax. don't know whether that makes sense. probably set to 0. anyways.
class MultiHeadSelfAttention2d(nn.Module):
    def __init__(self, num_heads: int, dim_head: int, dropout: float = 0., device: Union[str, torch.device] = None, dtype: torch.dtype = None) -> None:
        """
        Initializes multi head self-attention 2D module.

        Args:
            num_heads (int): Number of parallel self-attention heads.
            dim_head (int): Embedding dimensionality per self-attention head.
            dropout (float, optional): Dropout probability. Defaults to 0..
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.embed_dim = self.dim_head * self.num_heads

        self.in_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None, need_attn_maps: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs self-attention on 2D data.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output data [B, E, L, D]; attention maps [B, H, E*L, E*L] (optional).
        """

        B, E, L, D = x.size()
        assert D == self.embed_dim

        attn_mask = None
        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool
            assert padding_mask.size() == (B, E, L)
            padding_mask = padding_mask.view(B, 1, 1, 1, E, L).expand(-1, self.num_heads, -1, -1, -1, -1).reshape(B, self.num_heads, 1, 1, E, L)
            attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(padding_mask, -10000)

        q, k, v = self.in_projection(x).chunk(3, dim=-1)  # [B, E, L, D]
        q = q.view(B, E * L, self.num_heads, self.dim_head)  # [B, E * L, H, DH]
        k = k.view(B, E * L, self.num_heads, self.dim_head)
        v = v.view(B, E * L, self.num_heads, self.dim_head)

        q = q * self.dim_head ** -0.5
        attn = torch.einsum('bihc,bjhc->bhij', q, k)  # [B, H, E*L, E*L]
        if attn_mask is not None:
            attn = attn.view(B, self.num_heads, E, L, E, L)
            attn += attn_mask
            attn = attn.view(B, self.num_heads, E * L, E * L)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bjhd->bihd', attn, v)  # [B, E*L, H, DH]
        out = out.contiguous()
        out = out.view(B, E, L, D)  # [B, E, L, D]
        if need_attn_maps:
            return out, attn
        return out


class AxialSelfAttention2d(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_head: int,
                 dropout: float = 0.,
                 layer_norm_eps: float = 1e-5,
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes multi head axial self-attention 2D module.

        Args:
            num_heads (int): Number of parallel axial self-attention heads.
            dim_head (int): Embedding dimensionality per axial self-attention head.
            dropout (float, optional): Dropout probability. Defaults to 0..
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

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

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                need_attn_maps: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs axial self-attention on 2D data.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.

        Returns:
            Union[torch.Tensor,
                Tuple[torch.Tensor, torch.Tensor]]: Output batch_data [B, E, L, D];
                row attention maps [B, H, E, L, L] (optional).
        """

        B, E, L, D = x.size()
        assert D == self.embed_dim

        attn_mask = None
        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool
            assert padding_mask.size() == (B, E, L)
            padding_mask = padding_mask.view(B, 1, E, L).expand(-1, self.num_heads, -1, -1)
            attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(padding_mask, -10000)  # [B, H, E, L]

        q, k, v = self.in_row_projection(x).chunk(3, dim=-1)  # [B, E, L, D]
        q = q.view(B, E, L, self.num_heads, self.dim_head)  # [B, E, L, H, DH]
        k = k.view(B, E, L, self.num_heads, self.dim_head)
        v = v.view(B, E, L, self.num_heads, self.dim_head)

        q = q * self.dim_head ** -0.5
        # NOTE row attn
        row_attn = torch.einsum('bsihc, bsjhc->bhsij', q, k)  # [B, H, E, L, L]
        if attn_mask is not None:
            row_attn_mask = attn_mask.view(B, self.num_heads, E, 1, L).expand(-1, -1, -1, L, -1)
            row_attn += row_attn_mask
        row_attn = row_attn.softmax(dim=-1)
        row_attn = self.dropout1(row_attn)
        row_out = torch.einsum('bhsij, bsjhd->bsihd', row_attn, v)

        # NOTE: view not possible here, need reshape
        row_out = row_out.reshape(B, E, L, D)
        out = x + row_out
        out = self.norm1(out)

        q, k, v = self.in_col_projection(out).chunk(3, dim=-1)  # [B, E, L, D]
        q = q.view(B, E, L, self.num_heads, self.dim_head)  # [B, E, L, H, DH]
        k = k.view(B, E, L, self.num_heads, self.dim_head)
        v = v.view(B, E, L, self.num_heads, self.dim_head)

        q = q * self.dim_head ** -0.5
        # NOTE col attn
        col_attn = torch.einsum('bilhc, bjlhc->bhijl', q, k)  # [B, H, E, E, L]
        if attn_mask is not None:
            col_attn_mask = attn_mask.view(B, self.num_heads, 1, E, L).expand(-1, -1, E, -1, -1)
            col_attn += col_attn_mask
        col_attn = col_attn.softmax(dim=-2)
        col_attn = self.dropout2(col_attn)
        col_out = torch.einsum('bhijl, bjlhd->bilhd', col_attn, v)

        # NOTE: view not possible here, need reshape
        col_out = col_out.reshape(B, E, L, D)

        out = out + col_out
        out = self.norm2(out)

        if need_attn_maps:
            return out, row_attn
        return out


# NOTE difference to original tied axial attention: Row attention done first, to have something akin to a learned positional embedding along the evolutionary axis
class TiedAxialSelfAttention2d(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_head: int,
                 dropout: float = 0.,
                 layer_norm_eps: float = 1e-5,
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes multi head tied axial self-attention 2D module.

        Args:
            num_heads (int): Number of parallel axial self-attention heads.
            dim_head (int): Embedding dimensionality per axial self-attention head.
            dropout (float, optional): Dropout probability. Defaults to 0..
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

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

    def row_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor,
                 need_attn_maps: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs row attention.

        Args:
            q (torch.Tensor): Query tensor [B, E, L, H, DH].
            k (torch.Tensor): Key tensor [B, E, L, H, DH].
            v (torch.Tensor): Value tensor [B, E, L, H, DH].
            attn_mask (torch.Tensor): Attention mask for padded elements [B, H, E, L].
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output data [B, E, L, D]; row attention maps [B, H, L, L] (optional).
        """

        B, E, L, _, _ = q.size()
        row_attn_maps = torch.einsum('bsihc, bsjhc->bhij', q, k)  # [B, H, L, L]
        if attn_mask is not None:
            row_attn_mask = attn_mask.sum(-2).view(B, self.num_heads, 1, L).expand(-1, -1, L, -1)  # [B, H, L, L]
            row_attn_maps += row_attn_mask
        row_attn_maps = row_attn_maps.view(B, self.num_heads, 1, L, L)
        row_attn_maps = row_attn_maps.softmax(dim=-1)  # [B, H, 1, L, L]
        row_attn_maps = self.dropout1(row_attn_maps)
        row_out = torch.einsum('bhsij, bsjhc->bsihc', row_attn_maps, v)  # [B, E, L, H, DH]
        row_out = row_out.reshape(B, E, L, self.embed_dim)  # [B, E, L, D]
        if need_attn_maps:
            return row_out, row_attn_maps
        return row_out

    def col_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor,
                 need_attn_maps: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs column attention.

        Args:
            q (torch.Tensor): Query tensor [B, E, L, H, DH].
            k (torch.Tensor): Key tensor [B, E, L, H, DH].
            v (torch.Tensor): Value tensor [B, E, L, H, DH].
            attn_mask (torch.Tensor): Attention mask for padded elements [B, H, E, L].
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output data [B, E, L, D]; column attention maps [B, H, L, L] (optional).
        """

        B, E, L, _, _ = q.size()
        col_attn_maps = torch.einsum('bilhc, bjlhc->bhijl', q, k)  # [B, H, E, E, L]
        if attn_mask is not None:
            col_attn_mask = attn_mask.view(B, self.num_heads, 1, E, L).expand(-1, -1, E, -1, -1)
            col_attn_maps += col_attn_mask
        col_attn_maps = col_attn_maps.softmax(dim=-2)
        col_attn_maps = self.dropout2(col_attn_maps)
        col_out = torch.einsum('bhijl, bjlhc->bilhc', col_attn_maps, v)  # [B, E, L, H, DH]
        col_out = col_out.reshape(B, E, L, self.embed_dim)  # [B, E, L, D]
        if need_attn_maps:
            return col_out, col_attn_maps
        return col_out

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                need_attn_maps: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs tied axial self-attention on 2D data.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            Output data [B, E, L, D]; row attention maps [B, H, 1, L, L] (optional).
        """

        B, E, L, D = x.size()
        assert D == self.embed_dim

        attn_mask = None
        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool
            assert padding_mask.size() == (B, E, L)
            padding_mask = padding_mask.view(B, 1, E, L).expand(-1, self.num_heads, -1, -1)
            attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(padding_mask, -10000)  # [B, H, E, L]

        q, k, v = self.in_row_projection(x).chunk(3, dim=-1)  # [B, E, L, D]
        q = q.view(B, E, L, self.num_heads, self.dim_head)  # [B, E, L, H, DH]
        k = k.view(B, E, L, self.num_heads, self.dim_head)
        v = v.view(B, E, L, self.num_heads, self.dim_head)

        q = q * (self.dim_head * E) ** -0.5
        # NOTE row attn
        if need_attn_maps:
            row_out, row_attn_maps = self.row_attn(q, k, v, attn_mask, need_attn_maps)
        else:
            row_out = self.row_attn(q, k, v, attn_mask, need_attn_maps)

        out = x + row_out
        out = self.norm1(out)

        q, k, v = self.in_col_projection(out).chunk(3, dim=-1)  # [B, E, L, D]
        q = q.view(B, E, L, self.num_heads, self.dim_head)  # [B, E, L, H, DH]
        k = k.view(B, E, L, self.num_heads, self.dim_head)
        v = v.view(B, E, L, self.num_heads, self.dim_head)

        q = q * self.dim_head ** -0.5
        # NOTE col attn
        if need_attn_maps:
            col_out, col_attn_maps = self.col_attn(q, k, v, attn_mask, need_attn_maps)
        else:
            col_out = self.col_attn(q, k, v, attn_mask, need_attn_maps)

        out = out + col_out
        out = self.norm2(out)

        if need_attn_maps:
            return out, row_attn_maps
        return out
    
    
class FastSelfAttention2d(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_head: int,
                 dropout: float = 0.,
                 layer_norm_eps: float = 1e-5,
                 num_features: int = 256,
                 use_hyperbolic: bool = False,
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes multi head tied axial self-attention 2D module.

        Args:
            num_heads (int): Number of parallel axial self-attention heads.
            dim_head (int): Embedding dimensionality per axial self-attention head.
            dropout (float, optional): Dropout probability. Defaults to 0..
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            num_features (int, optional): Number of random features. Defaults to 256.
            use_hyperbolic (bool, optional): Whether hyperbolic variant should be used. Defaults to False.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FastSelfAttention2d, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.embed_dim = self.dim_head * self.num_heads

        self.in_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)
        self.norm = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(p=dropout)
        
        self.num_features = num_features
        self.register_buffer("orf", self.create_orf(self.dim_head, self.num_features), persistent=False)
        self.apply_feature_map = self.apply_regular_feature_map
        if use_hyperbolic:
            self.apply_feature_map = self.apply_hyperbolic_feature_map

    @staticmethod
    def apply_scaling(scale, x):
        return torch.einsum("...n,...nd->...nd", scale, x)

    @staticmethod
    def create_orf(d_k, m):
        blocks = torch.randn(math.ceil(m / d_k), d_k, d_k)
        blocks, _ = torch.linalg.qr(blocks)
        scale = torch.randn(m, d_k).norm(dim=1)
        return FastSelfAttention2d.apply_scaling(scale, blocks.reshape(-1, d_k)[:m])

    @staticmethod
    def apply_regular_feature_map(x, orf, mask=0, epsilon=1e-6):
        m, d_k = orf.shape
        proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
        norm = (x ** 2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
        return (torch.exp(proj_x + mask - norm) + epsilon) / math.sqrt(m)

    @staticmethod
    def apply_hyperbolic_feature_map(x, orf, mask=0, epsilon=1e-6):
        m, d_k = orf.shape
        proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
        proj_x = torch.cat([proj_x, -proj_x], dim=-1)
        norm = (x ** 2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
        return (torch.exp(proj_x + mask - norm) + epsilon) / math.sqrt(2 * m)

    @staticmethod
    def fast_attention(query, key, value):
        buffer = torch.cat([key.transpose(1, 2).bmm(value), key.sum(1).unsqueeze(-1)], dim=-1)
        buffer = query.bmm(buffer)
        return FastSelfAttention2d.apply_scaling(1 / buffer[:, :, -1], buffer[:, :, :-1])

    def redraw_orf(self):
        m, d_k = self.orf.shape
        orf = self.create_orf(d_k, m)
        orf = orf.to(self.orf.device)
        self.register_buffer("orf", orf, persistent=False)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                need_attn_maps: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs tied axial self-attention on 2D data.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            Output data [B, E, L, D]; row attention maps [B, H, 1, L, L] (optional).
        """
        
        B, E, L, D = x.size()
        assert D == self.embed_dim

        attn_mask = None
        if padding_mask is not None:
            assert padding_mask.dtype == torch.bool
            assert padding_mask.size() == (B, E, L)
            padding_mask = padding_mask.view(B, 1, E, L).expand(-1, self.num_heads, -1, -1)
            attn_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(padding_mask, -10000)  # [B, H, E, L]
        
        q, k, v = self.in_projection(x).chunk(3, dim=-1)  # [B, E, L, D]
        q = q.view(B, E, L, self.num_heads, self.dim_head)  # [B, E, L, H, DH]
        k = k.view(B, E, L, self.num_heads, self.dim_head)
        v = v.view(B, E, L, self.num_heads, self.dim_head)
        
        attn_maps = None
        if need_attn_maps:
            attn_maps = q * (self.dim_head * E) ** -0.5
            attn_maps = torch.einsum('bsihc, bsjhc->bhsij', attn_maps, k)  # [B, H, E, L, L]
            if attn_mask is not None:
                attn_maps += attn_mask.view(B, self.num_heads, E, 1, L).expand(-1, -1, -1, L, -1)  # [B, H, E, L, L]
            attn_maps = attn_maps.softmax(dim=-1)  # [B, H, E, L, L]
            attn_maps = attn_maps.view(B, self.num_heads * E, L, L)  # [B, H * E, L, L]

        if attn_mask is not None:
            attn_mask = attn_mask.view(B * self.num_heads, E, L, 1).view(B * self.num_heads, E * L, 1).expand(-1, -1, self.num_features) # [B * H, E * L, M]
        else:
            attn_mask = 0
        
        q = q.view(B, E * L, self.num_heads, self.dim_head).permute(0, 2, 1, 3).view(B * self.num_heads, E * L, self.dim_head)  # [B * H, E * L, DH]
        k = k.view(B, E * L, self.num_heads, self.dim_head).permute(0, 2, 1, 3).view(B * self.num_heads, E * L, self.dim_head)  # [B * H, E * L, DH]
        v = v.view(B, E * L, self.num_heads, self.dim_head).permute(0, 2, 1, 3).view(B * self.num_heads, E * L, self.dim_head)  # [B * H, E * L, DH]
        
        # TODO: How to apply dropout?
        
        q = self.apply_feature_map(q, self.orf)  # [B * H, E * L, M]
        k = self.apply_feature_map(k, self.orf, mask=attn_mask)  # [B * H, E * L, M]
        out = self.fast_attention(q, k, v)  # [B * H, E * L, DH]
        out = out.view(B, self.num_heads, E * L, self.dim_head).view(B, self.num_heads, E, L, self.dim_head).permute(0, 2, 3, 1, 4).reshape(B, E, L, self.embed_dim) # [B, E, L, D]
        
        out = x + out
        out = self.norm(out)
        
        self.redraw_orf()
        
        if need_attn_maps:
            return out, attn_maps
        return out


class Transmorpher2d(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: int, norm: nn.Module = None) -> None:
        """
        Transmorpher 2D module, which is used as backbone for unsupervised learning on MSAs.

        Args:
            block (nn.Module): Block module, which is stacked.
            num_blocks (int): Number of blocks.
            norm (nn.Module, optional): Normalization module, which is applied in the end. Defaults to None.
        """

        super(Transmorpher2d, self).__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(block) for i in range(num_blocks)])
        self.num_blocks = num_blocks
        self.norm = norm

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                need_attn_maps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Passes input data through the self-attention based backbone, resulting in a latent representation.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
            Output data [B, E, L, D]; list of blocks' row attention maps (optional).
        """

        out = x
        if need_attn_maps:
            attns = []
            for block in self.blocks:
                out, a = block(out, padding_mask=padding_mask, need_attn_maps=need_attn_maps)
                attns.append(a)
            if self.norm is not None:
                out = self.norm(out)
            return out, attns

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask, need_attn_maps=need_attn_maps)
        if self.norm is not None:
            out = self.norm(out)

        return out


class TransmorpherBlock2d(nn.Module):
    def __init__(self,
                 dim_head: int,
                 num_heads: int,
                 dim_ff: int,
                 dropout: float = 0.1,
                 attention: str = 'tied',
                 activation: str = 'relu',
                 layer_norm_eps: float = 1e-5,
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes Transmorpher 2D block.

        Args:
            dim_head (int): Embedding dimensionality per self-attention head.
            num_heads (int): Number of parallel self-attention heads.
            dim_ff (int): Feedforward embedding dimensionality.
            attention (str): Used attention mechanism.
            activation (str): Used activation function.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        dim_model = dim_head * num_heads
        super(TransmorpherBlock2d, self).__init__()
        self.attn = _get_attention_function(attention)(num_heads, dim_head, dropout=dropout, **factory_kwargs)

        self.lin1 = nn.Linear(dim_model, dim_ff, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_ff, dim_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_function(activation)()

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                need_attn_maps: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Passes input data through the self-attention based block, resulting in a latent representation.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output data [B, E, L, D]; row attention maps (optional).
        """

        out = self.attn(x, padding_mask=padding_mask, need_attn_maps=need_attn_maps)
        if need_attn_maps:
            out, attn = out
        # TODO what's the last layer of an attention block? should it be a nonlinearity
        x = x + self.dropout1(out)
        x = self.norm1(out)
        out = self.lin2(self.dropout(self.activation(self.lin1(x))))
        x = x + self.dropout2(out)
        x = self.norm2(x)
        if need_attn_maps:
            return x, attn
        return x


def _get_attention_function(attention: str) -> Type[nn.Module]:
    """
    Returns the module that corresponds to the given attention mechanism.

    Args:
        attention (str): Attention mechanism. Currently implemented: full, axial, tied, fast.

    Raises:
        RuntimeError: Unknown attention mechanism.

    Returns:
        Type[nn.Module]: Attention module.
    """

    if attention == 'full':
        return MultiHeadSelfAttention2d
    elif attention == 'axial':
        return AxialSelfAttention2d
    elif attention == 'tied':
        return TiedAxialSelfAttention2d
    elif attention == 'fast':
        return FastSelfAttention2d
    raise RuntimeError("Expected full, axial, tied, or fast not {}".format(attention))


def _get_activation_function(activation: str) -> Type[nn.Module]:
    """
    Returns the module that corresponds to the given activation function.

    Args:
        activation (str): Activation function. Currently implemented: relu, gelu.

    Raises:
        RuntimeError: Unknown activation function.

    Returns:
        Type[nn.Module]: Activation function module.
    """

    if activation == 'relu':
        return nn.ReLU
    elif activation == 'gelu':
        return nn.GELU
    raise RuntimeError("Expected relu or gelu not {}".format(activation))
