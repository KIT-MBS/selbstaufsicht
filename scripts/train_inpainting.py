import os

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.transforms import MSA2Tensor, RandomMSAColumnMasking

# NOTE MSA transformer reference: num_layers=12, d=768, num_heads=12, batch_size=512, lr=10**-4, **-2 lr schedule, 32 V100 GPUs for 100k updates, finetune for 25k more

# training parameters
batch_size = 1
epochs = 1

# model parameters


root = os.environ['DATA_PATH'] + 'Xfam'
model = models.self_supervised.MSAModel()
transform = transforms.Compose([MSA2Tensor(rna2index), RandomMSAColumnMasking(p=0.15)])
ds = datasets.Xfam(root, download=True, transform=transform)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
trainer = Trainer(max_epochs=epochs)
trainer.fit(model, dl)
