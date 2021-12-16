import os
import subprocess as sp
from typing import Callable

import torch

from Bio import AlignIO

from .shrinked_force_permutations import ShrinkedForcePermutationsDataset


class ZwdDataset(ShrinkedForcePermutationsDataset):
    def __init__(self, root: str , transform: Callable = None):
        super().__init__(transform=transform)
        self.root = root
        
        if not os.path.isfile(root + '/not-for-Rfam'):
            sp.call(['git', 'clone', 'https://bitbucket.org/zashaw/zashaweinbergdata.git', f'{root}/'])

        withdrawn = set()
        with open(root + '/CHANGES.txt') as rf:
            for line in rf:
                line = line.split()
                if line != '':
                    modified_file = line[1]
                    modification = ' '.join(line[3:])
                    if 'withdrawn' in modification:
                        withdrawn.add(modified_file)

        with open(root + '/not-for-Rfam') as rf:
            for line in rf:
                line = line.strip()
                if line != '' and line not in withdrawn:
                    file_path = self.root + f'/{line}'
                    with open(file_path, 'rt', encoding='utf-8') as f:
                        self.samples.append(AlignIO.read(f, 'stockholm'))
