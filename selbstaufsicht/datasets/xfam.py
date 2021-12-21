import os
import gzip
from typing import Callable

from tqdm import tqdm

from Bio import AlignIO

from .shrinked_force_permutations import ShrinkedForcePermutationsDataset
from ._utils import get_family_ids, _download

splits = ['train', 'val', 'test']
polymers = {'rna': 'Rfam', 'protein': 'Pfam'}
modes = ['seed', 'enhanced', 'full']


class XfamDataset(ShrinkedForcePermutationsDataset):
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
            exclude_ids: set = set(),
            download: bool = False) -> None:
        super().__init__(transform=transform)
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
        self.exclude_ids = exclude_ids

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
        self.samples = [a for (i, a) in enumerate(self.samples) if self.fam_ids[i] not in self.exclude_ids]

        if self.mode == 'enhanced':
            print('load extended msas')
            for i, fam_id in enumerate(tqdm(self.fam_ids)):
                if fam_id in self.exclude_ids:
                    continue
                full_msa_path = os.path.join(self.root, self.base_folder, version, 'full', split, f'{fam_id}.sto')
                if os.path.isfile(full_msa_path):

                    try:
                        with open(full_msa_path, 'rt', encoding='utf-8') as f:
                            self.samples[i] = AlignIO.read(f, 'stockholm')
                    except ValueError:
                        print(fam_id)
        
        self._init_num_data_samples()
