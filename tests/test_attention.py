import torch
from torch.nn.functional import multi_head_attention_forward
import torch.testing as testing

from selbstaufsicht.modules import AxialLayerNorm, AxialSelfAttention2d, MultiHeadSelfAttention2d, TiedAxialSelfAttention2d, Transmorpher2d, TransmorpherLayer2d


def test_MultiHeadAttention2d():
    num_heads = 2
    dim_head = 3
    bs = 1
    embed_dim = num_heads * dim_head
    S = 2
    L = 3
    dropout = 0.

    module = MultiHeadSelfAttention2d(num_heads, dim_head)
    x = torch.rand(bs, embed_dim, S, L)

    key_padding_mask = None
    in_proj_weight = module.in_projection.weight.view(3 * embed_dim, embed_dim)
    in_proj_bias = module.in_projection.bias.view(3 * embed_dim)
    bias_k = None
    bias_v = None
    add_zero_attn = False
    out_proj_weight = torch.eye(embed_dim)
    out_proj_bias = torch.zeros(embed_dim)
    attn_mask = None

    pred, attn = module(x)

    x_ref = x.reshape(bs, embed_dim, S * L)
    x_ref = x_ref.permute(2, 0, 1)
    ref, ref_attn = multi_head_attention_forward(
        x_ref, x_ref, x_ref,
        embed_dim, num_heads, in_proj_weight, in_proj_bias,
        bias_k, bias_v, add_zero_attn, dropout, out_proj_weight, out_proj_bias,
        need_weights=True, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

    pred = pred.permute(2, 3, 0, 1).reshape_as(ref)

    testing.assert_allclose(pred, ref)
    testing.assert_allclose(attn.sum(dim=1) / num_heads, ref_attn)


# NOTE: Comparison of AxialAttention and TiedAxialAttention does not work, since SumReduce(Softmax(x)) != Softmax(SumReduce(x))
def test_AxialAttention2d():
    bs, h, w = 1, 2, 2
    num_heads, dim_head = 2, 2
    dim = num_heads * dim_head
    x = torch.arange(0, bs * dim * h * w,
                     dtype=torch.float).reshape(bs, dim, w, h)

    model = AxialSelfAttention2d(num_heads, dim_head)
    out, (row_attn, col_attn) = model(x)

    out_ref = torch.tensor([[[[-0.9868, -0.9865],
                              [-0.9504, -0.9503]],
                             [[-1.0013, -1.0016],
                              [-1.0323, -1.0324]],
                             [[1.1482,  1.1479],
                              [1.1726,  1.1725]],
                             [[0.8399,  0.8402],
                              [0.8101,  0.8102]]]])
    row_attn_ref = torch.tensor([[[[[0.1167, 0.8833],
                                    [0.0816, 0.9184]],
                                   [[0.0565, 0.9435],
                                    [0.0387, 0.9613]]],
                                  [[[0.9645, 0.0355],
                                    [0.9852, 0.0148]],
                                   [[0.9939, 0.0061],
                                    [0.9975, 0.0025]]]]])
    col_attn_ref = torch.tensor([[[[[0.4999, 0.4999],
                                    [0.5001, 0.5001]],
                                   [[0.4999, 0.4999],
                                    [0.5001, 0.5001]]],
                                  [[[0.4782, 0.4782],
                                    [0.5218, 0.5218]],
                                   [[0.4782, 0.4782],
                                    [0.5218, 0.5218]]]]])

    testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(row_attn, row_attn_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(col_attn, col_attn_ref, atol=1e-4, rtol=1e-3)


def test_TiedAxialAttention2d():
    bs, h, w = 1, 2, 2
    num_heads, dim_head = 2, 2
    dim = num_heads * dim_head
    x = torch.arange(0, bs * dim * h * w,
                     dtype=torch.float).reshape(bs, dim, w, h)

    model = TiedAxialSelfAttention2d(num_heads, dim_head)
    out, (row_attn, col_attn) = model(x)

    out_ref = torch.tensor([[[[-0.9859, -0.9858],
                              [-0.9501, -0.9500]],
                             [[-1.0023, -1.0023],
                              [-1.0326, -1.0327]],
                             [[1.1477,  1.1477],
                              [1.1723,  1.1723]],
                             [[0.8404,  0.8404],
                              [0.8103,  0.8103]]]])
    row_attn_ref = torch.tensor([[[[[7.8421e-03, 9.9216e-01],
                                    [3.5674e-03, 9.9643e-01]]],
                                  [[[9.9977e-01, 2.2546e-04],
                                    [9.9996e-01, 3.7457e-05]]]]])
    col_attn_ref = torch.tensor([[[[[0.4999, 0.4999],
                                    [0.5001, 0.5001]],
                                   [[0.4999, 0.4999],
                                    [0.5001, 0.5001]]],
                                  [[[0.4783, 0.4783],
                                    [0.5217, 0.5217]],
                                   [[0.4783, 0.4783],
                                    [0.5217, 0.5217]]]]])

    testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(row_attn, row_attn_ref, atol=1e-4, rtol=1e-3)
    testing.assert_allclose(col_attn, col_attn_ref, atol=1e-4, rtol=1e-3)


def test_Transmorpher():
    bs, h, w = 1, 2, 2
    num_heads, dim_head = 2, 2
    dim = num_heads * dim_head
    layer_norm_eps = 1e-5
    x = torch.arange(0, bs * dim * h * w,
                     dtype=torch.float).reshape(bs, dim, w, h)

    transmorpher_layer = TransmorpherLayer2d(
        dim_head, num_heads, 2 * dim_head * num_heads, attention='tied', activation='relu', layer_norm_eps=layer_norm_eps)
    transmorpher_norm = AxialLayerNorm(
        1, dim_head * num_heads, eps=layer_norm_eps)
    transmorpher = Transmorpher2d(transmorpher_layer, 2, transmorpher_norm)
    out = transmorpher(x)

    out_ref = torch.tensor([[[[-0.7612, -0.7343],
                              [-0.7261, -0.6907]],
                             [[-1.0568, -0.7807],
                              [-0.8974, -0.9911]],
                             [[ 1.4796,  1.6805],
                              [ 1.6332,  1.5914]],
                             [[ 0.3385, -0.1655],
                              [-0.0097,  0.0904]]]])
    
    testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-3)


# NOTE mostly done as exercise/affirmation
def test_linconfconsistency():
    bs, h, w = 1, 3, 3
    dim = 4
    dout = 3 * dim
    xconv = torch.arange(0, bs * dim * h * w,
                         dtype=torch.float).reshape(bs, dim, w, h)
    xlin = xconv.permute(0, 2, 3, 1)
    weight = torch.rand(dout, dim)
    bias = torch.rand(dout)
    lres = torch.nn.functional.linear(xlin, weight, bias)
    cres = torch.nn.functional.conv2d(
        xconv, weight.view(dout, dim, 1, 1), bias)

    testing.assert_allclose(cres, lres.permute(0, 3, 1, 2))


# TODO: Complete test
def test_padding_mask_axial():
    pass
