import os
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.utils import collate_msas_explicit_position

num_layers = 2
num_heads = 2
dim = 16
n_sequences = 100


root = os.environ['DATA_PATH']
model = models.self_supervised.MSAModel(
        num_layers=num_layers,
        num_heads=num_heaads,
        dim=dim,
        num_sequences=num_sequences)
dataset = datasets.Xfam(root, download=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_msas_explicit_position, num_workers=2)
trainer = Trainer(gpus=1, max_epochs=1)
trainer.fit(model, dataloader)
