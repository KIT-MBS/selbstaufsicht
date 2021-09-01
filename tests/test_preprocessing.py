import torch

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding


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
    masking = RandomMSAMasking(p=1., mode='token', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x = tokenize(alignment)
    assert x == torch.tensor()

    x = masking(x)
    masked, inpainting_target, inpainting_mask = x

    assert False

    x = positional(x)
