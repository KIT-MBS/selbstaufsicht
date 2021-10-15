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
def basic_msa():
    return MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
            SeqRecord(Seq("AAU.CUA"), id='seq2'),
            SeqRecord(Seq("CCUACU."), id='seq3'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
        ]
    )


@pytest.fixture
def tokenized_msa(basic_msa):
    tokenize = MSATokenize(rna2index)
    return tokenize(basic_msa)