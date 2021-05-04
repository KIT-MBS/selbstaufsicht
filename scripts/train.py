import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.utils import collate_msas


root = os.environ['DATA_PATH']
model = models.self_supervised.msa.MSAModel()
dataset = datasets.Xfam(root, download=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_msas, num_workers=8)
trainer = Trainer()
trainer.fit(model, dataloader)
