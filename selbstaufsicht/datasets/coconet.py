import os
import subprocess as sp

from torch.utils.data import Dataset

from Bio import AlignIO
from Bio.PDB.PDBParser import PDBParser


# TODO implement other contacts modes
# TODO implement contact extraction
# TODO implement fam_ids for train test split
class CocoNet(Dataset):
    def __init__(self, root, download=True, contacts='all', fam_ids=None):
        self.root = root
        assert contacts == 'all'
        if download:
            if not os.path.isfile(root + '/CocoNet/README.md'):
                sp.call(['git', 'clone', 'https://github.com/KIT-MBS/RNA-dataset.git', f'{root}/CocoNet/'])

        msa_files = os.listdir(root + '/CocoNet/MSA/')

        self.fam_ids = set([filename.split('.')[0].split('_')[0] for filename in msa_files])
        print(len(self.fam_ids))

        self.msa2pdb = {}
        self.msas = []
        if fam_ids is not None:
            raise ValueError('selection of families not yet implemented')
        for filename in msa_files:
            with open(self.root + '') as f:
                msa = AlignIO.read(f, 'fasta')
                self.msas.append(msa)

        self.pdbs = []
        for msa in self.msas:
            pdb_id = msa[0].id
            self.msa2pdb[filename.split('.')[0]] = pdb_id
            pdb_file = self.root + f'/CocoNet/PDB/{pdb_id}.pdb'
            with open(pdb_file, 'r') as f:
                self.pdbs.append(PDBParser())

    def __getitem__(self, i):
        return self.msas[i], self.pdbs[i]

    def __len__(self):
        return(len(self.msas))
