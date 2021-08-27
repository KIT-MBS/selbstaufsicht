import torch

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.utils import rna2index
from selbstaufsicht.model.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding


def test_msa_mask_token():
    torch.manual_seed(42)

    alignment = MultipleSeqAlignment(
            [
                SeqRecord(Seq("ACTCCTA"), id='seq1'),
                SeqRecord(Seq("AAT.CTA"), id='seq2'),
                SeqRecord(Seq("CCTACT."), id='seq3'),
                SeqRecord(Seq("TCTCCTC"), id='seq4'),
            ]
            )

    tokenize = MSATokenize(rna2index)
    masking = RandomMSAMasking(p=1., mode='token', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x = tokenize(alignment)
    assert x == torch.tensor()

    x = masking(x)
    masked, inpainting_target, inpainting_mask = x

    assert

    x = positional(x)

