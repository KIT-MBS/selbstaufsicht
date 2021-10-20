import torch
import torch.testing as testing

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.modules import InpaintingHead
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding, RandomMSASubsampling


def test_msa_tokenize(tokenized_msa):
    tokenized_msa_ref = torch.tensor([[3, 5, 4, 5, 5, 4, 3],
                                      [3, 3, 4, 1, 5, 4, 3],
                                      [5, 5, 4, 3, 5, 4, 1],
                                      [4, 5, 4, 5, 5, 4, 5]])
    testing.assert_equal(tokenized_msa, tokenized_msa_ref)


def test_msa_mask_token(tokenized_msa):
    masking = RandomMSAMasking(
        p=.5, mode='token', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x, y = masking(tokenized_msa)
    x, y = positional((x, y))

    x_ref = {'aux_features': torch.tensor([[[0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000]],
                                           [[0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571]]]),
             'msa': torch.tensor([[ 3,  5, 18,  5, 18,  4, 18],
                                  [ 3,  3, 18,  1,  5,  4,  3],
                                  [ 5, 18,  4,  3, 18,  4, 18],
                                  [18, 18,  4, 18, 18, 18, 18]]),
             'mask': torch.tensor([[False, False,  True, False,  True, False,  True],
                                   [False, False,  True, False, False, False, False],
                                   [False,  True, False, False,  True, False,  True],
                                   [ True,  True, False,  True,  True,  True,  True]])}

    y_ref = {'inpainting': torch.tensor(
        [4, 5, 3, 4, 5, 5, 1, 4, 5, 5, 5, 4, 5])}

    testing.assert_allclose(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_equal(x['msa'], x_ref['msa'])
    testing.assert_equal(x['mask'], x_ref['mask'])
    testing.assert_equal(y['inpainting'], y_ref['inpainting'])


def test_msa_mask_column(tokenized_msa):
    masking = RandomMSAMasking(
        p=.5, mode='column', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x, y = masking(tokenized_msa)
    x, y = positional((x, y))

    x_ref = {'aux_features': torch.tensor([[[0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000]],
                                           [[0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571]]]),
             'msa': torch.tensor([[18, 5, 18, 5, 18, 4, 18],
                                  [18, 3, 18, 1, 18, 4, 18],
                                  [18, 5, 18, 3, 18, 4, 18],
                                  [18, 5, 18, 5, 18, 4, 18]]),
             'mask': torch.tensor([[True, False, True, False, True, False, True],
                                   [True, False, True, False, True, False, True],
                                   [True, False, True, False, True, False, True],
                                   [True, False, True, False, True, False, True]])}

    y_ref = {'inpainting': torch.tensor(
        [3, 4, 5, 3, 3, 4, 5, 3, 5, 4, 5, 1, 4, 4, 5, 5])}

    testing.assert_allclose(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_equal(x['msa'], x_ref['msa'])
    testing.assert_equal(x['mask'], x_ref['mask'])
    testing.assert_equal(y['inpainting'], y_ref['inpainting'])


def test_msa_mask_block(tokenized_msa):
    masking = RandomMSAMasking(
        p=.5, mode='block', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x, y = masking(tokenized_msa)
    x, y = positional((x, y))

    x_ref = {'aux_features': torch.tensor([[[0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000]],
                                           [[0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571]]]),
             'msa': torch.tensor([[3, 5, 18, 18, 18, 4, 3],
                                  [3, 3, 18, 18, 18, 4, 3],
                                  [5, 5, 18, 18, 18, 4, 1],
                                  [4, 5, 18, 18, 18, 4, 5]]),
             'mask': torch.tensor([[False, False, True, True, True, False, False],
                                   [False, False, True, True, True, False, False],
                                   [False, False, True, True, True, False, False],
                                   [False, False, True, True, True, False, False]])}

    y_ref = {'inpainting': torch.tensor(
        [4, 5, 5, 4, 1, 5, 4, 3, 5, 4, 5, 5])}

    testing.assert_allclose(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_equal(x['msa'], x_ref['msa'])
    testing.assert_equal(x['mask'], x_ref['mask'])
    testing.assert_equal(y['inpainting'], y_ref['inpainting'])


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
    
    testing.assert_allclose(out, out_ref, atol=1e-4, rtol=1e-3)


def test_subsampling(basic_msa):
    sampler = RandomMSASubsampling(3, False, 'uniform')
    sampled = sampler(basic_msa)

    sampled_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("CCUACU."), id='seq3'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
        ]
    )

    for idx in range(len(sampled)):
        assert sampled[idx].seq == sampled_ref[idx].seq
