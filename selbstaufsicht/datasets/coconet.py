import os
import re
import subprocess as sp
import pathlib

from torch.utils.data import Dataset

from Bio import AlignIO
from Bio.PDB.PDBParser import PDBParser


class CoCoNetDataset(Dataset):
    """
    CoCoNet: contact prediction dataset used in: https://doi.org/10.1093/nar/gkab1144
    to maximize confusion and the amount of test data, we use the test dataset to
    train the downstream 'unsupervised' contact prediction and the train dataset for
    testing
    """
    def __init__(self, root, split, transform=None, download=True):
        self.root = pathlib.Path(root)
        self.transform = transform
        if download:
            if not os.path.isfile(root + '/coconet/README.md'):
                sp.call(['git', 'clone', 'https://github.com/KIT-MBS/coconet.git', f'{root}/coconet/'])

        # NOTE this is intentionally swapped. in our case we want a large test set and a small train set
        split_dir = 'RNA_DATASET' if split == 'test' else 'RNA_TESTSET'
        msa_index_filename = 'CCNListOfMSAFiles.txt'

        with open(pathlib.Path(self.root / 'coconet' / split_dir / msa_index_filename), 'rt') as f:
            self.fam_ids = [line.strip() for line in f]

        self.msas = []
        for fam_id in self.fam_ids:
            with open(self.root / 'coconet' / split_dir / 'MSA' / (fam_id + '.faclean')) as f:
                msa = AlignIO.read(f, 'fasta')
                self.msas.append(msa)

        self.pdbs = []
        self.pdb_ids = []
        for msa in self.msas:
            pdb_id = re.split(r'.\|', (msa[0].id))[-2].strip('_.').lower()

            self.pdb_ids.append(pdb_id)
            pdb_path = self.root / 'coconet' / split_dir / 'PDBFiles' / (pdb_id + '.pdb')
            with open(pdb_path, 'r') as f:
                self.pdbs.append(PDBParser().get_structure(pdb_id, f))

    def __getitem__(self, i):
        x = self.msas[i]
        y = self.pdbs[i]
        if self.transform is not None:
            return self.transform({'msa': x}, {'structure': y})

        return x, y

    def __len__(self):
        return(len(self.msas))
