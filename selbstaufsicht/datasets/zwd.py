import os
import subprocess as sp

from torch.utils.data import Dataset

from Bio import AlignIO


class zwd(Dataset):
    def __init__(self, root):
        self.root = root
        if not os.path.isfile(root + '/zwd/not-for-Rfam'):
            sp.call(['git', 'clone', 'https://bitbucket.org/zashaw/zashaweinbergdata.git', f'{root}/zwd/'])
        self.sample_files = []

        withdrawn = set()
        with open(root + '/zwd/CHANGES.txt') as f:
            for line in f:
                line = line.split()
                if line != '':
                    modified_file = line[1]
                    modification = ' '.join(line[3:])
                    if 'withdrawn' in modification:
                        withdrawn.add(modified_file)

        with open(root + '/zwd/not-for-Rfam') as f:
            for line in f:
                line = line.strip()
                if line != '' and line not in withdrawn:
                    self.sample_files.append(line)

    def __getitem__(self, i):
        file_path = self.root + f'/zwd/{self.sample_files[i]}'
        with open(file_path, 'rt', encoding='utf-8') as f:
            sample = AlignIO.read(f, 'stockholm')
        return sample

    def __len__(self):
        return(len(self.sample_files))
