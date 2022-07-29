import torch
import torch.testing as testing

# from selbstaufsicht.models.self_supervised.msa.modules import InpaintingHead, JigsawHead, ContrastiveHead
from selbstaufsicht.models.self_supervised.msa.modules import InpaintingHead, JigsawHead


def test_inpainting_head():
    num_classes = 4
    bs, S, L, D = 2, 3, 4, 8

    model = InpaintingHead(D, num_classes)
    latent = torch.rand((bs, S, L, D))
    mask = torch.full((bs, S, L), 0.5)
    mask = torch.bernoulli(mask).to(torch.bool)

    out = model(latent, {'mask': mask})
    out_ref = torch.tensor([[0.8108, 0.4546,  0.0308, -0.3818],
                            [0.7133, 0.7049, -0.0218, -0.3852],
                            [0.6086, 0.4439, -0.2997, -0.0855],
                            [0.5279, 0.2488, -0.1313, -0.0970],
                            [0.4448, 0.5628, -0.2541, -0.0900],
                            [0.4254, 0.5415, -0.1067, -0.0165],
                            [0.3169, 0.5469, -0.2521,  0.0763],
                            [0.5263, 0.7497, -0.0420, -0.3318],
                            [0.4400, 0.8195, -0.1103,  0.0479],
                            [0.5831, 0.7132, -0.2069, -0.2395],
                            [0.8939, 0.5034,  0.1073, -0.3953],
                            [0.5738, 0.5081, -0.0580, -0.1092],
                            [0.6805, 0.5888, -0.1333, -0.3499],
                            [0.7092, 0.4774,  0.0135, -0.2209],
                            [0.5564, 0.4920, -0.2069, -0.0589],
                            [0.7314, 0.1561, -0.0317, -0.2996]])

    testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-3)


def test_jigsaw_head():
    num_classes = 4
    bs, S, L, D = 2, 3, 4, 8

    model = JigsawHead(D, num_classes)
    latent = torch.rand((bs, S, L, D))

    out = model(latent, None)
    out_ref = torch.tensor([[[0.8108,  0.4767,  0.5279],
                             [0.4546,  0.6611,  0.2488],
                             [0.0308, -0.2844, -0.1313],
                             [-0.3818, -0.1364, -0.0970]],
                            [[0.5761,  0.8939,  0.7092],
                             [0.1075,  0.5034,  0.4774],
                             [-0.0964,  0.1073,  0.0135],
                             [0.0473, -0.3953, -0.2209]]])

    testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-3)


# def test_contrastive_head():
#     bs, S, L, D = 2, 3, 4, 8
#
#     model = ContrastiveHead(D)
#     latent = torch.rand((bs, S, L, D))
#
#     out = model(latent, None)
#     out_ref = torch.tensor([[-0.0048, -0.3309, -0.0306, 0.1585, -0.3924, -0.3743, -0.1985, 0.4601],
#                             [0.0364, -0.2817, -0.0870, 0.1965, -0.3521, -0.2907, -0.1541, 0.5072]])
#
#     testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-3)
