import pytest
import torch

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize


@pytest.fixture(autouse=True)
def fix_seed():
    torch.manual_seed(42)
    yield


@pytest.fixture
def msa_sample():
    return ({'msa': MultipleSeqAlignment(
            [
                SeqRecord(Seq("ACUCCUA"), id='seq1'),
                SeqRecord(Seq("AAU.CUA"), id='seq2'),
                SeqRecord(Seq("CCUACU."), id='seq3'),
                SeqRecord(Seq("UCUCCUC"), id='seq4'),
            ]
            )},
            {})


@pytest.fixture
def tokenized_sample(msa_sample):
    tokenize = MSATokenize(rna2index)
    return tokenize(*msa_sample)
