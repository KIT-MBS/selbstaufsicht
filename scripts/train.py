import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

from selbstaufsicht import models
from selbstaufsicht import datasets

class LitMod(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = models.TransformerEncoderStack(4, 32, 4, 128, 12)
        self.crit = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # [x] == (l, s, b, n)
        pred = self(x)
        pred = nn.functional.log_softmax(x, dim=-1)
        loss = self.crit(y, pred)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters())

# TODO data module

# TODO optimize
def collate_msas(msas):
    B = len(msas)
    S = max([len(msa) for msa in msas])
    L = max([msa.get_alignment_length() for msa in msas])
    D = 4

    from selbstaufsicht.utils import rna_to_tensor_dict

    batch = torch.zeros((S, L, B, D), dtype=torch.float)

    for i, msa in enumerate(msas):
        for s in range(len(msa)):
            for l in range(msa.get_alignment_length()):
                batch[s, l, i, ...] = rna_to_tensor_dict[msa[s, l]][...]

    batch = batch.reshape(S*L, B, D)
    return (batch, batch)

root = os.environ['DATA_PATH']
model = LitMod()
dataset = datasets.Xfam(root, download=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_msas, num_workers=8)
trainer = Trainer()
trainer.fit(model, dataloader)
