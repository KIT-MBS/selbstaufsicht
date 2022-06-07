from torch.utils.data import Dataset

from Bio import AlignIO

from ..utils import rna2index


class InferenceDataset(Dataset):
    """Dataset for inference, without ground-truth"""
    def __init__(self, fasta_files, transform=None):
        self.transform = transform
        self.token_mapping = rna2index

        self.msas = []
        for fasta_file in fasta_files:
            with open(fasta_file) as f:
                msa = AlignIO.read(f, 'fasta')
                self.msas.append(msa)

    def __getitem__(self, i):
        x = self.msas[i]

        if self.transform is not None:
            return self.transform({'msa': x}, {'structure': None})

        return x, None

    def __len__(self):
        return(len(self.msas))