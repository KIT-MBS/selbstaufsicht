import gzip

from Bio import AlignIO
from torchtext.utils import download_from_url

splits = ['train', 'val', 'test']
polymers = {'rna' : 'Rfam', 'protein' : 'Pfam'}
modes = ['seed', 'full']

class Xfam():
    """
    Dataset for self-supervised learning based on the xfam databases.
    """

    def __init__(self, root,
            mode='seed',
            split='train',
            version='9.1',
            polymer='rna',
            download=False):
        assert split in splits
        assert mode in modes
        db = polymers[polymer]



    def __getitem__(self, i):
        return self.samples[i]
