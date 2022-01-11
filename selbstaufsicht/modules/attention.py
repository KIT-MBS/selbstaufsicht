import copy
from contextlib import ExitStack
import math
from typing import Any, List, Tuple, Type, Union

import torch
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils import checkpoint

from . import differentiable_functions as df

# TODO make c and d consistent
# TODO make E and H and L and W consistent
# TODO attention maps are formatted: [batch_size, num_heads, E, L], padding masks have to be adapted to that from [batch_size, E, L]


# TODO key padding masks
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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None, need_attn_maps: bool = True,
                attn_chunk_size: int = 0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs self-attention on 2D data.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.
            attn_chunk_size (int, optional): Chunk size in attention computation. Defaults to 0.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output data [B, E, L, D]; attention maps [B, H, E*L, E*L] (optional).
        """
        
        # TODO: Implement attn chunking

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
        # TODO optimize calls to contiguous
        out = out.contiguous()
        out = out.view(B, E, L, D)  # [B, E, L, D]
        if need_attn_maps:
            return out, attn
        return out


class AxialSelfAttention2d(nn.Module):
    # TODO optimize einsum strings using opt_einsum package
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
                need_attn_maps: bool = True,
                attn_chunk_size: int = 0) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs axial self-attention on 2D data.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.
            attn_chunk_size (int, optional): Chunk size in attention computation. Defaults to 0.

        Returns:
            Union[torch.Tensor,
                Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: Output batch_data [B, E, L, D];
                row attention maps [B, H, E, L, L] (optional); column attention maps [B, H, E, E, L] (optional).
        """
        
        # TODO: Implement attn chunking

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
            return out, (row_attn, col_attn)
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
        
        self.dropout2_chunking = df.Dropout(p=dropout)
        self.softmax_chunking = df.Softmax()
    
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
    
    class ColAttnChunked(Function):
        """Implements a chunked version of tied axial column attention."""
        
        @staticmethod
        def chunk_col_attn(q_chunked: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor, dropout: df.DifferentiableModule, 
                           softmax: df.DifferentiableModule, autocast: bool = False, no_backward: bool = False) -> torch.Tensor:
            """
            Computes a chunk of the column attention maps.

            Args:
                q_chunked (torch.Tensor): Chunked query tensor [B, EC, L, H, DH].
                k (torch.Tensor): Key tensor [B, E, L, H, DH].
                attn_mask (torch.Tensor): Attention mask for padded elements [B, H, E, L].
                dropout (df.DifferentiableModule): Manually differentible dropout module.
                softmax (df.DifferentiableModule): Manually differentible softmax module.
                autocast (bool, optional): Whether autocast is active. Defaults to False.
                no_backward (bool, optional): Whether computational context should not be cached for a subsequent backward pass. Defaults to False.

            Returns:
                torch.Tensor: Chunked column attention maps [B, H, EC, E, L].
            """
            
            B, EC, L, H, _ = q_chunked.size()
            _, E, _, _, _ = k.size()
            grad_mode = torch.is_grad_enabled()
            torch.set_grad_enabled(grad_mode and not no_backward)
            col_attn_maps = torch.einsum('bilhc, bjlhc->bhijl', q_chunked, k)  # [B, H, EC, E, L]
            if attn_mask is not None:
                col_attn_mask = attn_mask.view(B, H, 1, E, L).expand(-1, -1, EC, -1, -1)
                col_attn_maps += col_attn_mask
            col_attn_maps = softmax(col_attn_maps, dim=-2, no_backward=no_backward)
            col_attn_maps = dropout(col_attn_maps, autocast=autocast, no_backward=no_backward)
            torch.set_grad_enabled(grad_mode)
            return col_attn_maps
        
        @staticmethod
        @custom_fwd
        def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor, dropout: df.DifferentiableModule, 
                    softmax: df.DifferentiableModule, chunk_size: int) -> torch.Tensor:
            """
            Performs forward pass of the chunked tied axial column attention.

            Args:
                ctx (Any): Computational context.
                q (torch.Tensor): Query tensor [B, E, L, H, DH].
                k (torch.Tensor): Key tensor [B, E, L, H, DH].
                v (torch.Tensor): Value tensor [B, E, L, H, DH].
                attn_mask (torch.Tensor): Attention mask for padded elements [B, H, E, L].
                dropout (df.DifferentiableModule): Manually differentible dropout module.
                softmax (df.DifferentiableModule): Manually differentible softmax module.
                chunk_size (int): Chunk size.

            Returns:
                torch.Tensor: Output data [B, E, L, D].
            """
            
            assert chunk_size > 0
            ctx.save_for_backward(q, k, v, attn_mask)
            
            B, E, L, H, DH = q.size()
            E_chunked = min(E, chunk_size)
            num_chunks = E // E_chunked
            E_rest = E % E_chunked
            if E_rest > 0:
                num_chunks += 1
            
            ctx.dropout = dropout
            ctx.softmax = softmax
            ctx.B = B
            ctx.E = E
            ctx.L = L
            ctx.H = H
            ctx.DH = DH
            ctx.num_chunks = num_chunks
            ctx.E_chunked = E_chunked
            ctx.E_rest = E_rest
            
            # preserve rng states, cf. https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = False
            ctx.had_autocast_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
                # q, k, v, attn_mask are intended to be on the same device, so only check q
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = checkpoint.get_device_states(q)
            
            col_out = torch.empty((B, E, L, H, DH), device=q.device)
            # create chunks over evolutionary query dim, which is not reduced afterwards
            for idx in range(num_chunks):
                chunk_slice = (slice(None,), slice(idx*E_chunked, (idx+1)*E_chunked), slice(None,), slice(None,), slice(None,))
                col_attn_maps_chunk = TiedAxialSelfAttention2d.ColAttnChunked.chunk_col_attn(q[chunk_slice], k, attn_mask, dropout, 
                                                                                             softmax, autocast=ctx.had_autocast_in_fwd,
                                                                                             no_backward=True)  # [B, H, EC, E, L]
                col_out[chunk_slice] = torch.einsum('bhijl, bjlhc->bilhc', col_attn_maps_chunk, v)  # [B, EC, L, H, DH]
                del col_attn_maps_chunk
            col_out = col_out.reshape(B, E, L, H*DH)  # [B, E, L, D]
            
            return col_out

        @staticmethod
        @custom_bwd
        def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
            """
            Performs backward pass of the chunked tied axial column attention.

            Args:
                ctx (Any): Computational context.
                grad_out (torch.Tensor): Incoming derivative w.r.t. the output.

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]: Derivatives w.r.t. the queries, keys, and values; 
                dummy arguments, required by the autograd engine.
            """
            
            q, k, v, attn_mask = ctx.saved_tensors
            
            grad_dtype = torch.float16 if ctx.had_autocast_in_fwd else torch.float32
            q_grad = torch.empty((ctx.B, ctx.E, ctx.L, ctx.H, ctx.DH), dtype=grad_dtype, device=q.device)
            k_grad = torch.zeros((ctx.B, ctx.E, ctx.L, ctx.H, ctx.DH), dtype=grad_dtype, device=k.device)
            v_grad = torch.zeros((ctx.B, ctx.E, ctx.L, ctx.H, ctx.DH), dtype=grad_dtype, device=v.device)
            
            # preserve rng states, cf. https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
            rng_devices = []
            if ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices):
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    checkpoint.set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                q, k, v = checkpoint.detach_variable((q, k, v))
                v = v.permute(0, 2, 3, 4, 1)                                                # [B, L, H, DH, E]
                v = v.reshape(ctx.B * ctx.L * ctx.H, ctx.DH, ctx.E)                         # [B*L*H, DH, E]
                
                # create chunks over evolutionary query dim, which was not reduced in the 
                # forward pass
                for idx in range(ctx.num_chunks):
                    if idx == ctx.num_chunks - 1 and ctx.E_rest > 0:
                        EC = ctx.E_rest
                    else:
                        EC = ctx.E_chunked
                    
                    chunk_slice = (slice(None,), slice(idx*ctx.E_chunked, (idx+1)*ctx.E_chunked), slice(None,), 
                                   slice(None,), slice(None,))
                    # since attention maps were not cached, they have to be re-computed
                    a = TiedAxialSelfAttention2d.ColAttnChunked.chunk_col_attn(q[chunk_slice], 
                                                                               k, attn_mask, 
                                                                               ctx.dropout, 
                                                                               ctx.softmax,
                                                                               autocast=ctx.had_autocast_in_fwd)  # [B, H, EC, E, L]
                    
                    # compute a_grad
                    temp = grad_out[chunk_slice[:-1]]                                       # [B, EC, L, D]
                    temp = temp.reshape(ctx.B, EC, ctx.L, ctx.H, ctx.DH)                    # [B, EC, L, H, DH]
                    temp = temp.permute(0, 2, 3, 1, 4)                                      # [B, L, H, EC, DH]
                    temp = temp.reshape(ctx.B * ctx.L * ctx.H, EC, ctx.DH)                  # [B*L*H, EC, DH]
                    a_grad = temp @ v                                                       # [B*H*L, EC, E]
                    a_grad = a_grad.reshape(ctx.B, ctx.L, ctx.H, EC, ctx.E)                 # [B, L, H, EC, E]
                    a_grad = a_grad.permute(0, 2, 3, 4, 1)                                  # [B, H, EC, E, L]
                    a_grad = ctx.dropout.backward(a_grad)                                   # [B, H, EC, E, L]
                    a_grad = ctx.softmax.backward(a_grad)                                   # [B, H, EC, E, L]
                    
                    # compute v_grad
                    temp = temp.permute(0, 2, 1)                                            # [B*L*H, DH, EC]
                    a = a.permute(0, 4, 1, 2, 3)                                            # [B, L, H, EC, E]
                    a = a.reshape(ctx.B * ctx.L * ctx.H, EC, ctx.E)                         # [B*L*H, EC, E]
                    temp = temp @ a                                                         # [B*L*H, DH, E]
                    temp = temp.reshape(ctx.B, ctx.L, ctx.H, ctx.DH, ctx.E)                 # [B, L, H, DH, E]
                    temp = temp.permute(0, 4, 1, 2, 3)                                      # [B, E, L, H, DH]
                    v_grad += temp
                    
                    # compute q_grad
                    a_grad = a_grad.permute(0, 4, 1, 2, 3)                                  # [B, L, H, EC, E]
                    a_grad = a_grad.reshape(ctx.B * ctx.L * ctx.H, EC, ctx.E)               # [B*L*H, EC, E]
                    temp = k.permute(0, 2, 3, 1, 4)                                         # [B, L, H, E, DH]
                    temp = temp.reshape(ctx.B * ctx.L * ctx.H, ctx.E, ctx.DH)               # [B*L*H, E, DH]
                    temp = a_grad @ temp                                                    # [B*L*H, EC, DH]
                    temp = temp.reshape(ctx.B, ctx.L, ctx.H, EC, ctx.DH)                    # [B, L, H, EC, DH]
                    temp = temp.permute(0, 3, 1, 2, 4)                                      # [B, EC, L, H, DH]
                    q_grad[chunk_slice] = temp
                    
                    # compute k_grad 
                    a_grad = a_grad.permute(0, 2, 1)                                        # [B*L*H, E, EC]
                    temp = q[chunk_slice].permute(0, 2, 3, 1, 4)                            # [B, L, H, EC, DH]
                    temp = temp.reshape(ctx.B * ctx.L * ctx.H, EC, ctx.DH)                  # [B*L*H, EC, DH]
                    temp = a_grad @ temp                                                    # [B*L*H, E, DH]
                    temp = temp.reshape(ctx.B, ctx.L, ctx.H, ctx.E, ctx.DH)                 # [B, L, H, E, DH]
                    temp = temp.permute(0, 3, 1, 2, 4)                                      # [B, E, L, H, DH]
                    k_grad += temp

                    del a
                    del a_grad
                    del temp

            return q_grad, k_grad, v_grad, None, None, None, None

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                need_attn_maps: bool = True,
                attn_chunk_size: int = 0) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs tied axial self-attention on 2D data.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to True.
            attn_chunk_size (int, optional): Chunk size in attention computation. Defaults to 0.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
            Output data [B, E, L, D]; row attention maps [B, H, L, L] (optional); column attention maps [B, H, E, E, L] (optional).
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
        if attn_chunk_size > 0:
            col_out = self.ColAttnChunked.apply(q, k, v, attn_mask, self.dropout2_chunking, self.softmax_chunking, attn_chunk_size)
            if need_attn_maps:
                # TODO: not feasible, what to do here?
                col_attn_maps = None
        else:
            if need_attn_maps:
                col_out, col_attn_maps = self.col_attn(q, k, v, attn_mask, need_attn_maps)
            else:
                col_out = self.col_attn(q, k, v, attn_mask, need_attn_maps)
            

        out = out + col_out
        out = self.norm2(out)

        if need_attn_maps:
            return out, (row_attn_maps, col_attn_maps)
        return out


# TODO not sure about the name, self attention, transform oneself... mutate... morph meh
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

    # TODO need attn is a bit clunkily done
    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                need_attn_maps: bool = False,
                attn_chunk_size: int = 0) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]]:
        """
        Passes input data through the self-attention based backbone, resulting in a latent representation.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to False.
            attn_chunk_size (int, optional): Chunk size in attention computation. Defaults to 0.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]]:
            Output data [B, E, L, D]; list of blocks' attention maps (optional).
        """

        out = x
        if need_attn_maps:
            attns = []
            for block in self.blocks:
                out, a = block(out, padding_mask=padding_mask, need_attn_maps=need_attn_maps, attn_chunk_size=attn_chunk_size)
                attns.append(a)
            if self.norm is not None:
                out = self.norm(out)
            return out, attns

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask, need_attn_maps=need_attn_maps, attn_chunk_size=attn_chunk_size)
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
        self.attn = _get_attention_function(attention)(dim_head, num_heads, dropout=dropout, **factory_kwargs)

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
                need_attn_maps: bool = False,
                attn_chunk_size: int = 0) -> Union[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Passes input data through the self-attention based block, resulting in a latent representation.

        Args:
            x (torch.Tensor): Input data [B, E, L, D].
            padding_mask (torch.Tensor, optional): Padding mask [B, E, L]. Defaults to None.
            need_attn_maps (bool, optional): Whether attention maps should be returned. Defaults to False.
            attn_chunk_size (int, optional): Chunk size in attention computation. Defaults to 0.

        Returns:
            Union[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: Output data [B, E, L, D]; attention maps (optional).
        """

        out = self.attn(x, padding_mask=padding_mask, need_attn_maps=need_attn_maps, attn_chunk_size=attn_chunk_size)
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
        attention (str): Attention mechanism. Currently implemented: full, axial, tied.

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
    raise RuntimeError("Expected full, axial, or tied not {}".format(attention))


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
