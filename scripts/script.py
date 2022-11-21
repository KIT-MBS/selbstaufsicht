import pandas as pd
from Bio import SeqIO

import numpy as np

a=np.zeros((2413,100))
k=3
cv=10
seqs=[]
for i in range(cv*k):
    f=SeqIO.parse('/hkfs/work/workspace/scratch/qx6387-profile3/alina/alina/testset_aln_'+str(i)+'.fasta','fasta')
    for seq in f:
        seqs.append(seq.id)
    #f.close()
#seq1=[]
#for seq in f1:
#    seq1.append(int(seq.id))

#f2=SeqIO.parse('testset_aln_1.fasta','fasta')
#
#seq2=[]
#for seq in f2:
#    seq2.append(int(seq.id))

#f3=SeqIO.parse('testset_aln_2.fasta','fasta')

#seq3=[]
#for seq in f3:
#    seq3.append(int(seq.id))

out=np.loadtxt("output_1.txt")
i=0



#seqs=[*seq1,*seq2,*seq3]

for kk in range(cv):
    for k in seqs[kk*2413:(kk+1)*2413]:
        a[int(k),kk]=out[i]
        i=i+1

a=np.mean(a,axis=1)

df=pd.read_csv('/hkfs/work/workspace/scratch/qx6387-profile3/alina/alina/sample_submission.csv')

df=df.drop(['tm'],axis=1)

#df.insert(1,'tm',a*111.8,False)

df.insert(1,'tm',a*18.1+43.375,False)

df.to_csv("sample_submission4.csv",index=False)

