import os

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_tasks
from selbstaufsicht.utils import pad_collate_fn
from selbstaufsicht.utils import rna2index as token_dict

# training parameters
epochs = 1
# NOTE single GPU for now
batch_size = 512 // 32
lr = 0.0001
warmup = 16000
# TODO implement msa subsampling random, hamming maximizing
msa_sampling = 'random'

# model parameters
num_layers = 12
d = 768
num_heads = 12
d_head = d//num_heads
tasks = ['inpainting']


transform, heads = get_tasks(tasks, d, token_dict, subsampling=msa_sampling, masking_strategy='token')

root = os.environ['DATA_PATH'] + 'Xfam'
ds = datasets.Xfam(root, download=True, transform=transform)
model = models.self_supervised.MSAModel(num_layers, num_heads, d, heads=heads)
dl = DataLoader(ds, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)
trainer = Trainer(max_epochs=epochs)
trainer.fit(model, dl)
