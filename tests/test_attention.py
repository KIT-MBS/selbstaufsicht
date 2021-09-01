import torch
from selbstaufsicht.modules import MultiHeadSelfAttention2d, AxialSelfAttention2d, TiedAxialSelfAttention2d, Transmorpher2d
from torch.nn.functional import multi_head_attention_forward


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

    assert torch.allclose(pred, ref)
    assert torch.allclose(attn.sum(dim=1) / num_heads, ref_attn)


# TODO replace with tied axial attention consistency test
def test_AxialAttention2d():
    bs, h, w = 1, 5, 7
    dim = 8
    x = torch.arange(0, bs * dim * h * w, dtype=torch.float).reshape(bs, dim, w, h)

    model = AxialSelfAttention2d(2, 4)

    _, attn = model(x)

    tiedmodel = TiedAxialSelfAttention2d(2, 4)

    _, tiedattn = tiedmodel(x)

    assert (attn.sum() == tiedattn).all()


def test_Transmorpher():
    for attn in ['']:
        model = Transmorpher2d()

        x = torch.rand()
        model(x)


# NOTE mostly done as exercise/affirmation
def test_linconfconsistency():
    bs, h, w = 1, 3, 3
    dim = 4
    dout = 3 * dim
    xconv = torch.arange(0, bs * dim * h * w, dtype=torch.float).reshape(bs, dim, w, h)
    xlin = xconv.permute(0, 2, 3, 1)
    weight = torch.rand(dout, dim)
    bias = torch.rand(dout)
    lres = torch.nn.functional.linear(xlin, weight, bias)
    cres = torch.nn.functional.conv2d(xconv, weight.view(dout, dim, 1, 1), bias)

    assert torch.allclose(cres, lres.permute(0, 3, 1, 2))


# TODO
def test_padding_mask_axial():
    raise


if __name__ == '__main__':
    # test_MultiHeadAttention2d()
    test_AxialAttention2d()
    # test_linconfconsistency()
