import os
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms

root = os.environ['DATA_PATH']

ds = datasets.CoCoNetDataset(root, download=True, split='train', transform=get_downstream_transforms(400))

# xfam = datasets.XfamDataset(root, download=True, mode='seed', version='14.6')
# xfam_reduced = datasets.XfamDataset(root, download=True, mode='seed', version='14.6', exclude_ids=ds.fam_ids)

print(len(ds))
print(len(xfam))
print(len(xfam_reduced))

maxlen = 0
for x, y in ds:
    if x.get_alignment_length() > maxlen:
        maxlen = x.get_alignment_length()
    print('--------')
print(maxlen)
