import os
import gzip
import urllib
import urllib.request
from tqdm import tqdm

from Bio import AlignIO

splits = ['train', 'val', 'test']
polymers = {'rna' : 'Rfam', 'protein' : 'Pfam'}
modes = ['seed', 'full']

class Xfam():
    """
    Dataset for self-supervised learning based on the xfam family of biological sequence databases.
    """

    def __init__(self, root: str,
            mode: str ='seed',
            split: str ='train',
            version: str ='9.1',
            polymer: str ='rna',
            download: bool = False) -> None:
        if split not in splits: raise ValueError(f"split has to be in {splits}")
        if mode not in modes: raise ValueError(f"mode has to be in {modes}")
        if polymer not in polymers: ValueError(f"polymer has to be in {str(polymers)}")
        db = polymers[polymer]
        self.mode = mode
        self.split = split
        self.version = version
        self.root = root
        self.base_folder = db

        filename = db + '.gz'
        path = os.path.join(self.root, self.base_folder, version, mode, split, filename)
        if download:
            url = f'ftp://ftp.ebi.ac.uk/pub/databases/{db}/'+ f'{version}/{db}.{mode}.gz'
            self._download(url, path)
        if not os.path.isfile(path):
            raise RuntimeError('Dataset not found. Set download=True to download it.')
        with gzip.open(path, 'rt', encoding='latin1') as f:
            self.samples = [a for a in AlignIO.parse(f, 'stockholm')]

    def __getitem__(self, i):
        return self.samples[i]

    def _download(self, url, path):
        if os.path.isfile(path):
            print("Found existing dataset file.")
            return
        prefix, filename = os.path.split(path)
        os.makedirs(prefix, exist_ok=True)
        with open(path, 'wb') as f:
            with urllib.request.urlopen(urllib.request.FTPHandler) as response:
                with tqdm(total=response.length) as pbar:
                    for chunk in iter(lambda: reponse.read(chunk_size), ''):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        f.write(chunk)

    def __len__(self):
        return len(self.samples)
