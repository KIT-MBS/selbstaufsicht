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
             'msa': torch.tensor([[18, 18, 18,  5, 5,  4, 18],
                                  [18, 18,  4,  1, 5,  4,  3],
                                  [18, 18, 18, 18, 5, 18, 18],
                                  [18, 18, 18, 18, 5, 18,  5]]),
             'mask': torch.tensor([[True, True,  True, False, False, False,  True],
                                   [True, True, False, False, False, False, False],
                                   [True, True,  True,  True, False,  True,  True],
                                   [True, True,  True,  True, False,  True, False]])}

    y_ref = {'inpainting': torch.tensor(
        [3, 5, 4, 3, 3, 3, 5, 5, 4, 3, 4, 1, 4, 5, 4, 5, 4])}

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
             'msa': torch.tensor([[18, 18, 18, 18, 18, 18, 3],
                                  [18, 18, 18, 18, 18, 18, 3],
                                  [18, 18, 18, 18, 18, 18, 1],
                                  [18, 18, 18, 18, 18, 18, 5]]),
             'mask': torch.tensor([[True, True, True, True, True, True, False],
                                   [True, True, True, True, True, True, False],
                                   [True, True, True, True, True, True, False],
                                   [True, True, True, True, True, True, False]])}

    y_ref = {'inpainting': torch.tensor(
        [3, 5, 4, 5, 5, 4, 3, 3, 4, 1, 5, 4, 5, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4])}

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
             'msa': torch.tensor([[3, 18, 18, 18, 5, 4, 3],
                                  [3, 18, 18, 18, 5, 4, 3],
                                  [5, 18, 18, 18, 5, 4, 1],
                                  [4, 18, 18, 18, 5, 4, 5]]),
             'mask': torch.tensor([[False, True, True, True, False, False, False],
                                   [False, True, True, True, False, False, False],
                                   [False, True, True, True, False, False, False],
                                   [False, True, True, True, False, False, False]])}

    y_ref = {'inpainting': torch.tensor(
        [5, 4, 5, 3, 4, 1, 5, 4, 3, 5, 4, 5])}

    testing.assert_allclose(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_equal(x['msa'], x_ref['msa'])
    testing.assert_equal(x['mask'], x_ref['mask'])
    testing.assert_equal(y['inpainting'], y_ref['inpainting'])


# TODO: Complete test
def test_inpainting_head():
    pass
    # model = InpaintingHead
    # latent = torch.rand()
    # mask = torch.tensor()

    # out = model(latent, {'mask': mask})
    # assert out


def test_subsampling(basic_msa):
    sampler = RandomMSASubsampling(3, False, 'uniform')
    sampled = sampler(basic_msa)

    sampled_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
            SeqRecord(Seq("CCUACU."), id='seq3'),
        ]
    )

    for idx in range(len(sampled)):
        assert sampled[idx].seq == sampled_ref[idx].seq


if __name__ == '__main__':
    test_msa_mask_token()
