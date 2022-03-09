import numpy as np
import pytest
import torch

from Bio.PDB import Atom, Chain, Model, Residue, Structure
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


@pytest.fixture
def bio_structure():
    atom_coords = [
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ],
        [
            [0, 0, -1],
            [0, -2, 0],
            [-3, 0, 0]
        ],
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ],
        [
            [np.nan, np.nan, np.nan]
        ],
        [
            [0, 0, -4],
            [0, -5, 0],
            [-6, 0, 0]
        ],

    ]

    s_id = 0
    chain = Chain.Chain(0)
    for r_i in range(len(atom_coords)):
        residue = Residue.Residue(r_i, "%d" % r_i, 0)
        for a_i in range(len(atom_coords[r_i])):
            s_id += 1
            coords = np.array(atom_coords[r_i][a_i])
            name = "%d_%d" % (r_i, a_i)
            full_name = "%d_%d_%d" % (r_i, a_i, s_id)
            atom = Atom.Atom(name, coords, 42, 0, "altloc", full_name, s_id, "C")
            residue.add(atom)
        chain.add(residue)
    model = Model.Model(0)
    model.add(chain)
    structure = Structure.Structure(0)
    structure.add(model)
    return structure
