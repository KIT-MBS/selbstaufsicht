import os
import re
import subprocess as sp
import pathlib
from copy import deepcopy

import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from torch.utils.data import Dataset

from Bio import AlignIO
from Bio.PDB.PDBParser import PDBParser

from ..utils import rna2index


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
        self.token_mapping = rna2index
        if download:
            if not os.path.isfile(root + '/coconet/README.md'):
                sp.call(['git', 'clone', 'https://github.com/KIT-MBS/coconet.git', f'{root}/coconet/'])

        split_dir = 'RNA_DATASET' if split == 'train' else 'RNA_TESTSET'
        msa_index_filename = 'CCNListOfMSAFiles.txt'
        
        # NOTE: these MSAs are excluded, since they are problematic
        # too long sequences (400+)
        overlong_msa = {('3bwp', 'A'), ('4r0d', 'A')}
        # too few sequences (<50)
        small_msa = {('2krl', 'A'), ('2n1q', 'A'), ('2ke6', 'A'), ('4c4q', 'N')}
        # the hammerhead ribozyme somehow shows bad ppv performance, also in previous research using DCA methods 
        unknown_reason_msa = {('3zp8', 'A')}
        discarded_msa = overlong_msa | small_msa | unknown_reason_msa

        with open(pathlib.Path(self.root / 'coconet' / split_dir / msa_index_filename), 'rt') as f:
            self.fam_ids = [line.strip() for line in f]

        self.msas = []
        for fam_id in self.fam_ids:
            with open(self.root / 'coconet' / split_dir / 'MSA' / (fam_id + '.faclean')) as f:
                msa = AlignIO.read(f, 'fasta')
                self.msas.append(msa)

        pdb_index_filename = 'CCNListOfPDBFiles.txt'
        with open(pathlib.Path(self.root / 'coconet' / split_dir / pdb_index_filename), 'rt') as f:
            self.pdb_filenames = [line.strip() + '.pdb' for line in f]

        pdb_index_chain = 'CCNListOfPDBChains.txt'
        with open(pathlib.Path(self.root / 'coconet' / split_dir / pdb_index_chain), 'rt') as f:
            self.pdb_chains = [line.strip() for line in f]

        self.pdbs = []
        discarded_msa_idx = set()
        for idx, (msa, pdb_file, chain_id) in enumerate(zip(self.msas, self.pdb_filenames, self.pdb_chains)):
            refseq = msa[0].seq

            pdb_path = self.root / 'coconet' / split_dir / 'PDBFiles' / pdb_file
            with open(pdb_path, 'r') as f:
                pdb_id = pdb_file.split('_')[0]
                structure = PDBParser().get_structure(pdb_id, f)
                pdb_id = pdb_id.replace('.pdb', '')
                if (pdb_id, chain_id) in discarded_msa:
                    discarded_msa_idx.add(idx)
                    continue
                hetres = [r.get_id() for r in structure.get_residues() if r.get_id()[0] != ' ']
                # NOTE remove het residues
                chain = structure[0].get_list()[0]
                assert chain.get_id() == chain_id
                for id in hetres:
                    chain.detach_child(id)

                # NOTE fill in missing residues
                if len(refseq) > len(chain):
                    # NOTE there are missing residues
                    j = 0
                    resids = [r.get_id() for r in chain]
                    firstid = resids[0][1]
                    lastid = resids[-1][1]

                    # insert missing residues in the middle
                    dummy_atom = Atom(name='X',
                                      coord=np.array((np.nan, np.nan, np.nan), 'f'),
                                      bfactor=0.,
                                      occupancy=1.,
                                      altloc='',
                                      fullname='dummy',
                                      serial_number=0,
                                      element='C')

                    assert resids[-1][2] == ' '
                    for j, i in enumerate(range(firstid, lastid-1)):
                        if i not in chain:
                            r = Residue((' ', i, ' '), 'X', '')
                            r.add(deepcopy(dummy_atom))
                            chain.insert(j, r)

                # NOTE add missing residues at the ends
                if len(refseq) > len(chain):
                    matches = [m for m in re.finditer((''.join(r.get_resname() for r in chain)).replace('X', '.'), str(refseq))]
                    assert len(matches) == 1
                    assert matches[0].end() - matches[0].start() == len(chain)
                    # NOTE prefix
                    for i in range(matches[0].start()):
                        r = Residue((' ', i, ' '), 'X', '')
                        r.add(deepcopy(dummy_atom))
                        chain.insert(i, r)
                    # NOTE postfix
                    for i in range(matches[0].end(), len(refseq)):
                        r = Residue((' ', i, 'Y'), 'X', '')
                        r.add(deepcopy(dummy_atom))
                        chain.add(r)

                # NOTE fill in correct res names and renumber
                assert len(refseq) == len(chain)
                for i, r in enumerate(chain):
                    r.id = (' ', i+1, 'Z')
                    if r.get_resname() == 'X':
                        r.resname = refseq[i]
                for i, r in enumerate(chain):
                    r.id = (' ', i+1, ' ')

                self.pdbs.append(structure)
                assert refseq == ''.join([r.get_resname() for r in chain])
        assert len(self.msas) == len(self.fam_ids)

        # NOTE: Remove discarded MSAs
        self.msas = [msa for idx, msa in enumerate(self.msas) if idx not in discarded_msa_idx]

    def __getitem__(self, i):
        x = self.msas[i]
        y = self.pdbs[i]

        if self.transform is not None:
            return self.transform({'msa': x}, {'structure': y})

        return x, y

    def __len__(self):
        return(len(self.msas))
