import os
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.utils import collate_msas_explicit_position

num_layers = 2
num_heads = 2
num_heads = 4
num_sequences = 10
dim = 16

root = os.environ['DATA_PATH'] + 'Xfam'
# NOTE MSA transformer: num_layers=12, d=768, num_heads=12, batch_size=512, lr=10**-4, **-2 lr schedule, 32 V100 GPUs for 100k updates, finetune for 25k more
model = models.self_supervised.MSAModel(
        num_layers=num_layers,
        num_heads=num_heads,
        dim=dim,
        num_sequences=num_sequences)
dataset = datasets.Xfam(root, download=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_msas_explicit_position, num_workers=2)
trainer = Trainer(max_epochs=1)
trainer.fit(model, dataloader)
