import os
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_tasks, MSACollator


epochs = 20
batch_size = 2
lr = 0.0001
warmup = 16000

msa_sampling = 'token'

# model parameters
num_layers = 2
d = 768
num_heads = 12
d_head = d // num_heads
tasks = ['jigsaw', 'inpainting']

# TODO should take token mapping
transform, task_heads, task_losses, metrics = get_tasks(tasks, d)

root = os.environ['DATA_PATH'] + 'Xfam'
# NOTE MSA transformer: num_layers=12, d=768, num_heads=12, batch_size=512, lr=10**-4, **-2 lr schedule, 32 V100 GPUs for 100k updates, finetune for 25k more
ds = datasets.Xfam(root, download=True, transform=transform)
dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=MSACollator(), num_workers=6)
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
trainer = Trainer(max_epochs=epochs, gpus=1)
trainer.fit(model, dl)
