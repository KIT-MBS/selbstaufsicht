import os
import torch
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_downstream_transforms
from matplotlib import pyplot as plt

root = os.environ['DATA_PATH']

downstream_transform = get_downstream_transforms(subsample_depth=50, threshold=10., device='cpu')
ds = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform)

downstream_transform_3D = get_downstream_transforms(subsample_depth=50, threshold=10., device='cpu', secondary_window=2)
ds3D = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform_3D, secondary_window=4)

for i in range(len(ds)):
    print(ds.fam_ids[i], ds.pdb_filenames[i], ds.pdb_chains[i])
    x, y = ds[i]
    print(ds3D[i][1]['basepairs'])
    displaymap = y['contact'].triu() + ds3D[i][1]['contact'].tril()
    plt.imshow(displaymap)
    plt.colorbar()
    plt.show()
