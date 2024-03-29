import os
import re
import subprocess as sp
import pathlib
from copy import deepcopy

import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

import torch
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
    def __init__(
            self,
            root,
            split,
            transform=None,
            download=True,
            discard_train_size_based=True,
            diversity_maximization=False,
            max_seq_len: int = 400,
            min_num_seq: int = 50,
            secondary_window=-1):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.token_mapping = rna2index
        self.secondary_window = secondary_window
        if download:
            if not os.path.isfile(root + '/coconet/README.md'):
                sp.call(['git', 'clone', 'https://github.com/KIT-MBS/coconet.git', f'{root}/coconet/'])

        split_dir = 'RNA_DATASET' if split == 'train' else 'RNA_TESTSET'
        msa_index_filename = 'CCNListOfMSAFiles.txt'

        self.indices = None
        # TODO this file probably should go somewhere else and if it does not exist it should be created on demand if possible
        if diversity_maximization:
            indices_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coconet_%s_diversity_maximization.pt' % split)
            self.indices = torch.load(indices_path)

        # NOTE: these MSAs are excluded, since they are problematic
        # too long sequences
        self.max_seq_len = max_seq_len
        # too few sequences
        self.min_num_seq = min_num_seq
        # the hammerhead ribozyme somehow shows bad ppv performance, also in previous research using DCA methods
        # NOTE 2h0s is also problematic as it's technically a dimer
        discarded_msa = {('3zp8', 'A')}

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

                num_seq = len(msa)
                seq_len = msa.get_alignment_length()
                if (pdb_id, chain_id) in discarded_msa or ((split == 'test' or discard_train_size_based) and (num_seq < self.min_num_seq or seq_len > self.max_seq_len)):
                    print("Discarding MSA (pdb=%s, chain=%s)" % (pdb_id, chain_id))
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
        self.pdb_filenames = [x for idx, x in enumerate(self.pdb_filenames) if idx not in discarded_msa_idx]
        self.pdb_chains = [x for idx, x in enumerate(self.pdb_chains) if idx not in discarded_msa_idx]
        self.fam_ids = [x for idx, x in enumerate(self.fam_ids) if idx not in discarded_msa_idx]

        # NOTE load secondary structure data
        if self.secondary_window > -1:
            self.bprnaseqs = []
            self.bprnapairs = []
            for idx, pdb_file in enumerate(self.pdb_filenames):
                pdb_id = pdb_file.split('_')[0].replace('.pdb', '')
                bpseq_path = _get_bpseq_path(self.root, pdb_id, self.pdb_chains[idx])
                # print(self.fam_ids[idx], pdb_id, self.pdb_chains[idx])
                with open(bpseq_path, 'rt') as f:
                    lines = f.readlines()
                    offset = 0
                    while True:
                        letters = lines[offset + 1].strip().upper()
                        basepairs = lines[offset + 2].strip()
                        if self.pdb_chains[idx] == lines[offset].split('_')[1].strip():
                            break
                        offset += 3
                        assert offset < len(lines)
                    assert basepairs.count('(') == basepairs.count(')')
                    if self.msas[idx][0].seq != letters:
                        print(self.msas[idx][0].seq)
                        print(letters)
                    assert self.msas[idx][0].seq == letters
                    self.bprnaseqs.append(letters)
                    self.bprnapairs.append(basepairs)

    def __getitem__(self, i):
        x = self.msas[i]
        y = self.pdbs[i]

        if self.transform is not None:
            if self.indices is None:
                if self.secondary_window > -1:
                    item = self.transform({'msa': x}, {'structure': y, 'basepairs': self.bprnapairs[i]})
                else:
                    item = self.transform({'msa': x}, {'structure': y})
            else:
                if self.secondary_window > -1:
                    item = self.transform({'msa': x, 'indices': self.indices[i]}, {'structure': y, 'basepairs': self.bprnapairs[i]})
                else:
                    item = self.transform({'msa': x, 'indices': self.indices[i]}, {'structure': y})
            return item

        return x, y

    def __len__(self):
        return(len(self.msas))

def _get_bpseq_path(root, pdb_id, chain_id):
    bpseq_path = root / 'rnapdbee' / pdb_id.upper() / 'dssr-hybrid-isolated_removed-pseudoknots_as_unpaired_residues-no_image' / (pdb_id.upper() + '-2D-dotbracket.txt')

    if not os.path.isfile(bpseq_path):
        i = 1
        # TODO go through all folders
        while i < 10:
            bpseq_path = root / 'rnapdbee' / pdb_id.upper() / 'dssr-hybrid-isolated_removed-pseudoknots_as_unpaired_residues-no_image' / str(i) / (pdb_id.upper() + '-2D-dotbracket.txt')

            with open(bpseq_path, 'rt') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('>'):
                        if chain_id == line.split('_')[1].strip():
                            return bpseq_path
            i += 1
        if i < 10:
            print('NOTE: filtering out secondary contacts requires rnapdbee dotbracket files. Some of them have to be adapted, because in the coconet files some missing residues were removed.')
            raise

    return bpseq_path
