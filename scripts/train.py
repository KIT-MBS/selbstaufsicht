import os
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.utils import collate_msas_explicit_position


root = os.environ['DATA_PATH']
model = models.self_supervised.MSAModel(n_sequences=100)
dataset = datasets.Xfam(root, download=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_msas_explicit_position, num_workers=2)
trainer = Trainer(gpus=1, max_epochs=5)
trainer.fit(model, dataloader)
