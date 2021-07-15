import os
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.utils import collate_msas_explicit_position


root = os.environ['DATA_PATH']
# NOTE MSA transformer: num_layers=12, d=768, num_heads=12, batch_size=512, lr=10**-4, **-2 lr schedule, 32 V100 GPUs for 100k updates, finetune for 25k more
model = models.self_supervised.MSAModel(depth=8, heads=4, dim=32, n_sequences=10)
dataset = datasets.Xfam(root, download=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_msas_explicit_position, num_workers=2)
trainer = Trainer(max_epochs=1)
trainer.fit(model, dataloader)
