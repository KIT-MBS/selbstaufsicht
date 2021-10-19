import torch
from torch.nn.functional import multi_head_attention_forward
import torch.testing as testing

from selbstaufsicht.modules import AxialSelfAttention2d, MultiHeadSelfAttention2d, TiedAxialSelfAttention2d, Transmorpher2d, TransmorpherLayer2d


def test_MultiHeadAttention2d():
    num_heads = 2
    dim_head = 3
    bs = 1
    embed_dim = num_heads * dim_head
    S = 2
    L = 3
    dropout = 0.

    module = MultiHeadSelfAttention2d(num_heads, dim_head)
    x = torch.rand(bs, S, L, embed_dim)

    key_padding_mask = None
    in_proj_weight = module.in_projection.weight
    in_proj_bias = module.in_projection.bias
    bias_k = None
    bias_v = None
    add_zero_attn = False
    out_proj_weight = torch.eye(embed_dim)
    out_proj_bias = torch.zeros(embed_dim)
    attn_mask = None

    pred, attn = module(x)

    x_ref = x.reshape(bs, S * L, embed_dim)
    x_ref = x_ref.permute(1, 0, 2)
    ref, ref_attn = multi_head_attention_forward(
        x_ref, x_ref, x_ref,
        embed_dim, num_heads, in_proj_weight, in_proj_bias,
        bias_k, bias_v, add_zero_attn, dropout, out_proj_weight, out_proj_bias,
        need_weights=True, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

    pred = pred.permute(1, 2, 0, 3).reshape_as(ref)

    testing.assert_allclose(pred, ref)
    testing.assert_allclose(attn.sum(dim=1) / num_heads, ref_attn)


# NOTE: Comparison of AxialAttention and TiedAxialAttention does not work, since SumReduce(Softmax(x)) != Softmax(SumReduce(x))
# NOTE: Ref data used for comparison is the output of the current implementation (date: 10/19/2021)
def test_AxialAttention2d():
    bs, S, L = 1, 2, 2
    num_heads, dim_head = 2, 2
    embed_dim = num_heads * dim_head
    x = torch.arange(0, bs * embed_dim * S * L,
                     dtype=torch.float).reshape(bs, S, L, embed_dim)

    model = AxialSelfAttention2d(num_heads, dim_head)
    out, (row_attn, col_attn) = model(x)

    out_ref = torch.tensor([[[[-0.6975, -1.1449, 1.4239,  0.4186],
                              [-0.2929, -1.3020, 1.4856,  0.1094]],
                             [[ 0.2203, -1.3546, 1.4254, -0.2910],
                              [ 0.2643, -1.3218, 1.4298, -0.3723]]]])
    row_attn_ref = torch.tensor([[[[[8.0170e-02, 9.1983e-01],
                                    [1.5440e-04, 9.9985e-01]],
                                   [[2.7360e-07, 1.0000e+00],
                                    [4.8475e-10, 1.0000e+00]]],
                                  [[[8.1356e-01, 1.8644e-01],
                                    [1.0000e+00, 1.3278e-07]],
                                   [[1.0000e+00, 7.6927e-14],
                                    [1.0000e+00, 4.4570e-20]]]]])
    col_attn_ref = torch.tensor([[[[[0.5363, 0.5341],
                                    [0.4637, 0.4659]],
                                   [[0.5690, 0.5420],
                                    [0.4310, 0.4580]]],
                                  [[[0.2793, 0.3969],
                                    [0.7207, 0.6031]],
                                   [[0.3618, 0.4246],
                                    [0.6382, 0.5754]]]]])

    testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(row_attn, row_attn_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(col_attn, col_attn_ref, atol=1e-4, rtol=1e-3)


# NOTE: Ref data used for comparison is the output of the current implementation (date: 10/19/2021)
def test_TiedAxialAttention2d():
    bs, S, L = 1, 2, 2
    num_heads, dim_head = 2, 2
    embed_dim = num_heads * dim_head
    x = torch.arange(0, bs * embed_dim * S * L,
                     dtype=torch.float).reshape(bs, S, L, embed_dim)

    model = TiedAxialSelfAttention2d(num_heads, dim_head)
    out, (row_attn, col_attn) = model(x)

    out_ref = torch.tensor([[[[-0.2925, -1.3021, 1.4856,  0.1090],
                              [-0.2925, -1.3021, 1.4856,  0.1090]],
                             [[ 0.2643, -1.3218, 1.4298, -0.3723],
                              [ 0.2643, -1.3218, 1.4298, -0.3723]]]])
    row_attn_ref = torch.tensor([[[[[2.3846e-08, 1.0000e+00],
                                    [7.4856e-14, 1.0000e+00]]],
                                  [[[1.0000e+00, 1.7629e-14],
                                    [1.0000e+00, 5.9179e-27]]]]])
    col_attn_ref = torch.tensor([[[[[0.5341, 0.5341],
                                    [0.4659, 0.4659]],
                                   [[0.5419, 0.5419],
                                    [0.4581, 0.4581]]],
                                  [[[0.3970, 0.3970],
                                    [0.6030, 0.6030]],
                                   [[0.4247, 0.4247],
                                    [0.5753, 0.5753]]]]])

    testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(row_attn, row_attn_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(col_attn, col_attn_ref, atol=1e-4, rtol=1e-3)


# NOTE: Ref data used for comparison is the output of the current implementation (date: 10/19/2021)
def test_Transmorpher():
    bs, S, L = 1, 2, 2
    num_heads, dim_head = 2, 2
    embed_dim = num_heads * dim_head
    layer_norm_eps = 1e-5
    x = torch.arange(0, bs * embed_dim * S * L,
                     dtype=torch.float).reshape(bs, S, L, embed_dim)

    transmorpher_layer = TransmorpherLayer2d(
        dim_head, num_heads, 2 * dim_head * num_heads, attention='tied', activation='relu', layer_norm_eps=layer_norm_eps)
    transmorpher_norm = torch.nn.LayerNorm((embed_dim,))
    transmorpher = Transmorpher2d(transmorpher_layer, 2, transmorpher_norm)
    out = transmorpher(x)

    out_ref = torch.tensor([[[[-0.0348, -1.2462, 1.5420, -0.2610],
                              [ 0.2732, -1.1704, 1.4862, -0.5889]],
                             [[ 0.7663, -1.1246, 1.1999, -0.8415],
                              [ 0.0377, -1.1410, 1.5734, -0.4701]]]])

    testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-3)


# TODO: Complete test
def test_padding_mask_axial():
    pass
