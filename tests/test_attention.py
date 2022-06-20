import torch
from torch.nn.functional import multi_head_attention_forward
import torch.testing as testing

from selbstaufsicht.modules import AxialSelfAttention2d, MultiHeadSelfAttention2d, TiedAxialSelfAttention2d, Transmorpher2d, TransmorpherBlock2d
import selbstaufsicht.modules.differentiable_functions as df


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

    testing.assert_close(pred, ref)
    testing.assert_close(attn.sum(dim=1) / num_heads, ref_attn)


def test_padding_mask_MultiHeadAttention2d():
    num_heads = 2
    dim_head = 3
    bs = 2
    embed_dim = num_heads * dim_head
    S = 2
    L = 3

    pad_B = 0
    # must be stated as negative (inverse) index
    pad_S = -1
    pad_L = -1

    x = torch.rand(bs, S, L, embed_dim)
    module = MultiHeadSelfAttention2d(num_heads, dim_head)

    # test full padding in S dimension
    padding_mask = torch.zeros((bs, S, L), dtype=torch.bool)
    padding_mask[pad_B, :, pad_L] = True
    _, attn = module(x, padding_mask)
    attn_pad_ref = torch.zeros((num_heads, S * L))
    for s in range(1, S+1):
        attn_pad = attn[pad_B, :, :, s * L + pad_L]
        testing.assert_close(attn_pad, attn_pad_ref,
                             rtol=0, atol=0)

    # test full padding in L dimension
    padding_mask = torch.zeros((bs, S, L), dtype=torch.bool)
    padding_mask[pad_B, pad_S, :] = True
    _, attn = module(x, padding_mask)
    attn_pad_ref = torch.zeros((num_heads, S * L, L))
    attn_pad = attn[pad_B, :, :, (S + pad_S) * L: (S + pad_S + 1) * L]
    testing.assert_close(attn_pad, attn_pad_ref, rtol=0,
                         atol=0)

    # test full padding in S and L dimensions
    padding_mask = torch.zeros((bs, S, L), dtype=torch.bool)
    padding_mask[pad_B, pad_S, pad_L] = True
    _, attn = module(x, padding_mask)
    attn_pad_ref = torch.zeros((num_heads, S * L))
    attn_pad = attn[pad_B, :, :, (S + pad_S + 1) * L + pad_L]
    testing.assert_close(attn_pad, attn_pad_ref, rtol=0,
                         atol=0)


# NOTE: Comparison of AxialAttention and TiedAxialAttention does not work, since SumReduce(Softmax(x)) != Softmax(SumReduce(x))
# NOTE: Ref data used for comparison is the output of the current implementation (date: 10/19/2021)
def test_AxialAttention2d():
    bs, S, L = 1, 2, 2
    num_heads, dim_head = 2, 2
    embed_dim = num_heads * dim_head
    x = torch.arange(0, bs * embed_dim * S * L,
                     dtype=torch.float).reshape(bs, S, L, embed_dim)

    model = AxialSelfAttention2d(num_heads, dim_head)
    out, row_attn = model(x)

    out_ref = torch.tensor([[[[-0.8079, -1.0793, 1.3916,  0.4956],
                              [-0.2964, -1.2992, 1.4874,  0.1082]],
                             [[ 0.2213, -1.3585, 1.4227, -0.2856],
                              [ 0.2671, -1.3195, 1.4300, -0.3776]]]])
    row_attn_ref = torch.tensor([[[[[1.5118e-01, 8.4882e-01],
                                    [2.0143e-03, 9.9799e-01]],
                                   [[2.2873e-05, 9.9998e-01],
                                    [2.5922e-07, 1.0000e+00]]],
                                  [[[7.3919e-01, 2.6081e-01],
                                    [9.9999e-01, 1.3718e-05]],
                                   [[1.0000e+00, 5.3341e-10],
                                    [1.0000e+00, 2.0740e-14]]]]])

    testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-3)
    testing.assert_close(row_attn, row_attn_ref, atol=1e-4, rtol=1e-3)


def test_padding_mask_AxialAttention2d():
    num_heads = 2
    dim_head = 3
    bs = 2
    embed_dim = num_heads * dim_head
    S = 2
    L = 3

    pad_B = 0
    pad_S = -1
    pad_L = -1

    x = torch.rand(bs, S, L, embed_dim)
    module = AxialSelfAttention2d(num_heads, dim_head)

    # test full padding in S dimension
    padding_mask = torch.zeros((bs, S, L), dtype=torch.bool)
    padding_mask[pad_B, :, pad_L] = True
    _, row_attn = module(x, padding_mask)
    row_attn_pad = row_attn[pad_B, :, :, :, pad_L]
    row_attn_pad_ref = torch.zeros((num_heads, S, L))
    testing.assert_close(row_attn_pad, row_attn_pad_ref,
                         rtol=0, atol=0)

    # test full padding in S and L dimensions
    padding_mask = torch.zeros((bs, S, L), dtype=torch.bool)
    padding_mask[pad_B, pad_S, :] = True
    padding_mask[pad_B, :, pad_L] = True
    _, row_attn = module(x, padding_mask)
    row_attn_pad = row_attn[pad_B, :, :pad_S, :, pad_L]
    row_attn_pad_ref = torch.zeros((num_heads, S+pad_S, L))
    testing.assert_close(row_attn_pad, row_attn_pad_ref,
                         atol=0, rtol=0)


# NOTE: Ref data used for comparison is the output of the current implementation (date: 10/19/2021)
def test_TiedAxialAttention2d():
    bs, S, L = 1, 2, 2
    num_heads, dim_head = 2, 2
    embed_dim = num_heads * dim_head
    x = torch.arange(0, bs * embed_dim * S * L,
                     dtype=torch.float).reshape(bs, S, L, embed_dim)

    model = TiedAxialSelfAttention2d(num_heads, dim_head)
    out, row_attn = model(x)

    out_ref = torch.tensor([[[[-0.2907, -1.3007, 1.4875,  0.1039],
                              [-0.2902, -1.3009, 1.4875,  0.1035]],
                             [[ 0.2675, -1.3192, 1.4300, -0.3783],
                              [ 0.2678, -1.3192, 1.4299, -0.3785]]]])
    row_attn_ref = torch.tensor([[[[[1.5440e-04, 9.9985e-01],
                                    [2.7360e-07, 1.0000e+00]]],
                                  [[[1.0000e+00, 1.3277e-07],
                                    [1.0000e+00, 7.6928e-14]]]]])

    testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-3)
    testing.assert_close(row_attn, row_attn_ref, atol=1e-4, rtol=1e-3)


def test_padding_mask_TiedAxialAttention2d():
    num_heads = 2
    dim_head = 3
    bs = 2
    embed_dim = num_heads * dim_head
    S = 2
    L = 3

    pad_B = 0
    pad_S = -1
    pad_L = -1

    x = torch.rand(bs, S, L, embed_dim)
    module = TiedAxialSelfAttention2d(num_heads, dim_head)

    # test full padding in S dimension
    padding_mask = torch.zeros((bs, S, L), dtype=torch.bool)
    padding_mask[pad_B, :, pad_L] = True
    _, row_attn = module(x, padding_mask)
    row_attn_pad = row_attn[pad_B, :, 0, :, pad_L]
    row_attn_pad_ref = torch.zeros((num_heads, L))
    testing.assert_close(row_attn_pad, row_attn_pad_ref,
                         rtol=0, atol=0)

    # test full padding in S and L dimension
    padding_mask = torch.zeros((bs, S, L), dtype=torch.bool)
    padding_mask[pad_B, pad_S, :] = True
    padding_mask[pad_B, :, pad_L] = True
    _, row_attn = module(x, padding_mask)
    row_attn_pad = row_attn[pad_B, :, 0, :, pad_L]
    row_attn_pad_ref = torch.zeros((num_heads, L))
    
    testing.assert_close(row_attn_pad, row_attn_pad_ref,
                         rtol=0, atol=0)
    

# NOTE: Ref data used for comparison is the output of the current implementation (date: 10/19/2021)
# NOTE: Crashes in CI Job, but not locally. Reason unknown, so this test is omitted for now
# def test_Transmorpher():
#     bs, S, L = 1, 2, 2
#     num_heads, dim_head = 2, 2
#     embed_dim = num_heads * dim_head
#     layer_norm_eps = 1e-5
#     x = torch.arange(0, bs * embed_dim * S * L,
#                      dtype=torch.float).reshape(bs, S, L, embed_dim)

#     transmorpher_layer = TransmorpherBlock2d(
#         dim_head, num_heads, 2 * dim_head * num_heads, attention='tied', activation='relu', layer_norm_eps=layer_norm_eps)
#     transmorpher_norm = torch.nn.LayerNorm((embed_dim,))
#     transmorpher = Transmorpher2d(transmorpher_layer, 2, transmorpher_norm)
#     out = transmorpher(x)

#     out_ref = torch.tensor([[[[-0.0358, -1.2450, 1.5427, -0.2620],
#                               [ 0.2696, -1.1679, 1.4883, -0.5900]],
#                              [[ 0.7643, -1.1226, 1.2017, -0.8435],
#                               [ 0.0366, -1.1396, 1.5742, -0.4712]]]])

#     testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-3)
