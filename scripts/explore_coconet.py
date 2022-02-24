import os
import torch
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms

root = os.environ['DATA_PATH']
# TODO check thresholds
downstream_transform = get_downstream_transforms(subsample_depth=50, threshold=10., device='cpu')

ds = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform)

# xfam = datasets.XfamDataset(root, download=True, mode='seed', version='14.6')
# xfam_reduced = datasets.XfamDataset(root, download=True, mode='seed', version='14.6', exclude_ids=ds.fam_ids)

# print(len(ds))
# print(len(xfam))
# print(len(xfam_reduced))

maxlen = 0
pos = 0
total = 0
for x, y in ds:
    # if x['msa'].get_alignment_length() > maxlen:
    #     maxlen = x['msa'].get_alignment_length()
    pos += torch.sum(y['contact'] == 1)
    total += torch.sum(y['contact'] != -1)

print(pos)
print(total)
print(pos/total)
print(maxlen)
