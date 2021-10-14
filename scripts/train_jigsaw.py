import os

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from torchinfo import summary

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_tasks  # , MSACollator

# training parameters
epochs = 10
# NOTE single GPU for now
# batch_size = 512 // 32
batch_size = 1
lr = 0.0001
warmup = 16000
# TODO implement msa subsampling hamming maximizing

# model parameters
num_layers = 2
d = 768
num_heads = 12
d_head = d // num_heads
tasks = ['jigsaw']


# TODO should take token mapping
transform, task_heads, task_losses, metrics = get_tasks(tasks, d)

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
    metrics=metrics,
    in_dict_size=len(ds.token_mapping), padding_token=ds.token_mapping['PADDING_TOKEN']
)

summary(model)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
# dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=MSACollator())
print('run')
trainer = Trainer(max_epochs=epochs, gpus=1)
# trainer = Trainer(max_epochs=epochs)
trainer.fit(model, dl)
