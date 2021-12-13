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
            download: bool = False,
            debug_size: int = -1) -> None:
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

        if debug_size > 0:
            self.samples = self.samples[:debug_size]

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
                        ]),
                        MultipleSeqAlignment([
                            SeqRecord(Seq("AAAAAAAC"), id='eins'),
                            SeqRecord(Seq("AAAAAAAU"), id='zwei'),
                            SeqRecord(Seq("AAAAAAAG"), id='drei'),
                        ]),
                        MultipleSeqAlignment([
                            SeqRecord(Seq("AAAAAAAC"), id='eins'),
                            SeqRecord(Seq("--AAAA--"), id='zwei'),
                            SeqRecord(Seq("---AAAA-"), id='drei'),
                        ]),
                        MultipleSeqAlignment([
                            SeqRecord(Seq("AAAA--------CAUA"), id='eins'),
                            SeqRecord(Seq("AAAA--------CAGA"), id='zwei'),
                            SeqRecord(Seq("AAAA--------CA.A"), id='drei'),
                        ]),
                        ]

    def __getitem__(self, i):
        if self.transform is not None:
            return self.transform({'msa': self.samples[i]}, {})
        return self.samples[i]

    def __len__(self):
        return 1
