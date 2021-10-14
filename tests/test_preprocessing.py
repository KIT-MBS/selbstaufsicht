import torch
import torch.testing as testing

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.modules import InpaintingHead
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding, RandomMSASubsampling


def test_msa_mask_token():
    torch.manual_seed(42)

    alignment = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
            SeqRecord(Seq("AAU.CUA"), id='seq2'),
            SeqRecord(Seq("CCUACU."), id='seq3'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
        ]
    )

    tokenize = MSATokenize(rna2index)
    masking = RandomMSAMasking(
        p=1., mode='token', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x = tokenize(alignment)
    testing.assert_equal(x, torch.tensor([[3, 5, 4, 5, 5, 4, 3],
                                          [3, 3, 4, 1, 5, 4, 3],
                                          [5, 5, 4, 3, 5, 4, 1],
                                          [4, 5, 4, 5, 5, 4, 5]]))

    x, y = masking(x)
    x, y = positional((x, y))

    x_ref = {'aux_features': torch.tensor([[[0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000]],
                                           [[0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571]]]),
             'msa': torch.tensor([[18, 18, 18, 18, 18, 18, 18],
                                  [18, 18, 18, 18, 18, 18, 18],
                                  [18, 18, 18, 18, 18, 18, 18],
                                  [18, 18, 18, 18, 18, 18, 18]]),
             'mask': torch.tensor([[True, True, True, True, True, True, True],
                                  [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True],
                                   [True, True, True, True, True, True, True]])}

    y_ref = {'inpainting': torch.tensor([3, 5, 4, 5, 5, 4, 3,
                                         3, 3, 4, 1, 5, 4, 3,
                                         5, 5, 4, 3, 5, 4, 1,
                                         4, 5, 4, 5, 5, 4, 5])}

    testing.assert_allclose(x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
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


def test_subsampling():
    # TODO reproducibility
    alignment = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
            SeqRecord(Seq("AAU.CUA"), id='seq2'),
            SeqRecord(Seq("CCUACU."), id='seq3'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
        ]
    )

    sampler = RandomMSASubsampling(2, False, 'uniform')

    sampled = sampler(alignment)
    assert sampled


if __name__ == '__main__':
    test_msa_mask_token()
