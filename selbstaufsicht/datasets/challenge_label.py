import random
import os
import numpy as np
import re
import subprocess as sp
import pathlib
from copy import deepcopy
import glob
import numpy as np
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio import SeqIO

import torch
from torch.utils.data import Dataset

from Bio import AlignIO
from Bio.PDB.PDBParser import PDBParser

from ..utils import rna2index


class challData_lab(Dataset):
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
            discard_train_size_based=True,
            diversity_maximization=False,
            max_seq_len: int = 400,
            min_num_seq: int = 50,
            secondary_window=-1,num_cv=10,num_test=2413,test_splits=3):
        self.root = str(pathlib.Path(root))
        self.transform = transform
        self.token_mapping = rna2index
        self.secondary_window = secondary_window

        self.indices = None

        self.max_seq_len = max_seq_len
        self.min_num_seq = min_num_seq
        self.pdbs=[]
        self.msas = []
        self.num_cv=num_cv
        self.num_test=num_test
        self.test_splits=test_splits

        if split=='train':
            files=glob.glob(self.root+"/selbstaufsicht/selbstaufsicht/datasets/challenge_enzymes_fasta/*")
            for fam_id in files:
                with open(fam_id) as f:
                    msa = AlignIO.read(f, 'fasta')
                    self.msas.append(msa)
                nu=int(fam_id.split('/')[-1].split('group')[-1].split('.')[0])
                nu1=self.root+"/selbstaufsicht/selbstaufsicht/datasets/challenge_enzymes_labels/labels"+str(nu)+".txt"
                pdb=np.loadtxt(nu1,dtype=float)
                self.pdbs.append(pdb)
        else:
            f=self.root+"/selbstaufsicht/selbstaufsicht/datasets/testset_aln.fasta"
            
            fasta_seq=SeqIO.parse(open(f),'fasta')
            seqs=[]
            names=[]

            for fasta in fasta_seq:
                seqs.append(fasta.seq)
                names.append(fasta.id)
            
            for cv in range(self.num_cv):

                ind=np.random.permutation(num_test)

                #k=3

                for jj in range(self.test_splits):
                    ff=open(self.root+"/testset_aln_"+str(cv*self.test_splits+jj)+".fasta","w+")
                    
                    for jjj in range(jj*(self.num_test//self.test_splits+1),np.min([(jj+1)*(self.num_test//self.test_splits+1),self.num_test])):
                        ff.write(">"+names[ind[jjj]]+"\n")
                        ff.write(str(seqs[ind[jjj]])+"\n")
                    ff.close()

                for iii in range(self.test_splits):
                    ff=open(self.root+"/testset_aln_"+str(cv*self.test_splits+iii)+".fasta")
                    msa = AlignIO.read(ff, 'fasta')
                    self.msas.append(msa)
                    pdb=torch.zeros(len(msa))
                    self.pdbs.append(pdb)
                    ff.close()



    def __getitem__(self, i):
        x = self.msas[i]
        y = self.pdbs[i]/111.8
        y=np.float32(y)

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
        #    print(len(item[0]['msa']), item[1]['structure'].shape," item")
            return item
        return x, y

    def __len__(self):
        return(len(self.msas))

