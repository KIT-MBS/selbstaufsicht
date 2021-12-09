import os
import gzip
from typing import Callable

from tqdm import tqdm
import torch

from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from ._utils import get_family_ids, _download
from ..utils import rna2index

splits = ['train', 'val', 'test']
polymers = {'rna': 'Rfam', 'protein': 'Pfam'}
modes = ['seed', 'enhanced', 'full']


def getitem_modified(self, idx: int, num_data_samples: int, jigsaw_force_permutations: int):
    if self.transform is not None:
        if jigsaw_force_permutations:
            real_sample_idx = idx % num_data_samples
            permutation_idx = idx // num_data_samples
            sample = self.samples[real_sample_idx]
            num_seq = len(sample)
            labels = {'jigsaw': torch.full((num_seq,), permutation_idx, dtype=torch.int64)}
        else:
            sample = self.samples[idx]
            labels = {}
        return self.transform({'msa': sample}, labels)
    return self.samples[idx]

def len_modified(self, num_data_samples: int, jigsaw_force_permutations: int):
    if jigsaw_force_permutations:
        return min(len(self.samples), num_data_samples) * jigsaw_force_permutations
    else:
        return min(len(self.samples), num_data_samples)


class Xfam(torch.utils.data.Dataset):
    """
    Dataset for self-supervised learning based on the xfam family of biological sequence databases.
    """

    def __init__(
            self, root: str,
            mode: str = 'seed',
            split: str = 'train',
            version: str = '9.1',
            polymer: str = 'rna',
            transform: Callable = None,
            download: bool = False) -> None:
        if split not in splits:
            raise ValueError(f"split has to be in {splits}")
        if mode not in modes:
            raise ValueError(f"mode has to be in {modes}")
        if polymer not in polymers:
            raise ValueError(f"polymer has to be in {str(polymers)}")
        db = polymers[polymer]
        self.mode = mode
        self.split = split
        self.version = version
        self.root = root
        self.base_folder = db
        self.transform = transform
        self.token_mapping = rna2index

        if mode == 'full' and float(version) >= 12:
            raise ValueError('Starting with Rfam version 12.0 full alignments are no longer generated fully automatically.')
        if self.mode == 'enhanced':
            mode = 'seed'

        filename = db + '.gz'
        path = os.path.join(self.root, self.base_folder, version, mode, split, filename)
        if download:
            url = f'ftp://ftp.ebi.ac.uk/pub/databases/{db}/' + f'{version}/{db}.{mode}.gz'
            _download(url, path)
        if not os.path.isfile(path):
            raise RuntimeError('Dataset not found. Set download=True to download it.')
        self.fam_ids = get_family_ids(path)
        with gzip.open(path, 'rt', encoding='latin1') as f:
            self.samples = [a for a in AlignIO.parse(f, 'stockholm')]

        if self.mode == 'enhanced':
            print('load extended msas')
            for i, fam_id in enumerate(tqdm(self.fam_ids)):
                full_msa_path = os.path.join(self.root, self.base_folder, version, 'full', split, f'{fam_id}.sto')
                if os.path.isfile(full_msa_path):

                    try:
                        with open(full_msa_path, 'rt', encoding='utf-8') as f:
                            self.samples[i] = AlignIO.read(f, 'stockholm')
                    except ValueError:
                        print(fam_id)

    def __getitem__(self, i):
        if self.transform is not None:
            return self.transform({'msa': self.samples[i]}, {})
        return self.samples[i]

    def __len__(self):
        return len(self.samples)


class Dummy(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.token_mapping = rna2index
        self.transform = transform
        self.samples = [MultipleSeqAlignment([
                        SeqRecord(Seq("AAAACCCC"), id='eins'),
                        SeqRecord(Seq("AAAACCCC"), id='zwei'),
                        SeqRecord(Seq("AAAACCCC"), id='drei'),
                        ])]

    def __getitem__(self, i):
        if self.transform is not None:
            return self.transform({'msa': self.samples[i]}, {})
        return self.samples[i]

    def __len__(self):
        return len(self.samples)
