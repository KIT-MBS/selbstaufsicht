import os
import subprocess as sp
from typing import Callable
import glob
from Bio import AlignIO

from .shrinked_force_permutations import ShrinkedForcePermutationsDataset


class challDataset(ShrinkedForcePermutationsDataset):
    def __init__(self, root: str, 
            transform: Callable = None):
        super().__init__(transform=transform)
        self.root=root


        files=glob.glob(self.root+"/selbstaufsicht/selbstaufsicht/datasets/challenge_enzymes_fasta/*")
        for fi in files:
            self.samples.append(AlignIO.read(fi, 'fasta'))

        self._init_num_data_samples()
