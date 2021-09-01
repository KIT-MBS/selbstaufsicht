import os

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from torchinfo import summary

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_tasks
# from selbstaufsicht.utils import pad_collate_fn

# training parameters
epochs = 1
# NOTE single GPU for now
# batch_size = 512 // 32
batch_size = 1
lr = 0.0001
warmup = 16000
# TODO implement msa subsampling random, hamming maximizing
msa_sampling = 'token'

# model parameters
num_layers = 2
d = 768
num_heads = 12
d_head = d//num_heads
tasks = ['inpainting']


# TODO should take token mapping
transform, task_heads, task_losses = get_tasks(tasks, d, subsampling=msa_sampling, masking='token')

root = os.environ['DATA_PATH'] + 'Xfam'
print('data')
ds = datasets.Xfam(root, download=True, transform=transform)
print('model')
model = models.self_supervised.MSAModel(
        num_layers,
        num_heads,
        d_head,
        aux_input_dim=2,
        task_heads=task_heads,
        task_losses=task_losses,
        in_dict_size=len(ds.token_mapping), padding_token=ds.token_mapping['PADDING_TOKEN']
    )

summary(model)
# dl = DataLoader(ds, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
print('run')
trainer = Trainer(max_epochs=epochs)
trainer.fit(model, dl)
