import os
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.transforms import DistanceFromChain, ContactFromDistance

root = os.environ['DATA_PATH']

ds = datasets.CoCoNetDataset(root, download=True, split='train', target_transform=None)
t1 = DistanceFromChain()
t2 = ContactFromDistance()

xfam = datasets.XfamDataset(root, download=True, mode='seed', version='14.6')
xfam_reduced = datasets.XfamDataset(root, download=True, mode='seed', version='14.6', exclude_ids=ds.fam_ids)

print(len(ds))
print(len(xfam))
print(len(xfam_reduced))
