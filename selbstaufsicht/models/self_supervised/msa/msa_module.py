import math
import torch
from torch import nn

import pytorch_lightning as pl
from axial_attention import AxialAttention

from selbstaufsicht.modules import Transmorpher2d, TransmorpherLayer2d, AxialLayerNorm

# NOTE for using simCLR loss from bolts
# from pytorch_lightning.models.self_supervised.simclr.simclr_module import SyncFunction


class MSAModel(pl.LightningModule):
    """
    Model for pre-training on multiple sequence alignments of biological sequences
    """
    def __init__(
            self,
            num_layers=12,
            num_heads=12,
            dim_head=64,
            input_dim=30,
            attention='tied',
            activation='relu',
            layer_norm_eps=1e-5,
            task_heads=None,
            task_losses=None,
            device=None,
            dtype=None,
            ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        block = TransmorpherLayer2d(dim_head, num_heads, 2*dim_head*num_heads, attention=attention, activation=activation, layer_norm_eps=layer_norm_eps, **factory_kwargs)
        self.backbone = Transmorpher2d(block, num_layers, AxialLayerNorm(1, dim_head*num_heads, eps=layer_norm_eps, **factory_kwargs))
        self.tasks = [t for t in task_heads.keys()]
        self.heads = task_heads
        self.losses = task_losses
        assert self.heads.keys == self.losses.keys

    def forward(self, x, aux_features=None):
        """
        Forward pass through the model. Use for inference.
        Args:
            batch is a tuple of an input dict and a target dict
            the input dict contains a tokenized msa, any auxiliary features ('aux_features')
            and any additional for the task heads e.g. the mask where the loss is to be measured for the inpainting task.
            the output dict contains the target per task loss keyed per task
        """

        # NOTE feature dim = 1
        if aux_features is not None:
            x = torch.cat((self.embedding(x), aux_features), dim=1)
        latent = self.backbone(x)
        return latent

    def training_step(self, batch_data, batch_idx):
        x, y = batch_data

        latent = self.forward(x['msa'], x.get('aux_feautures', None))

        # TODO weights
        loss = sum([self.losses[task](self.heads['task'](latent, x), y['task']) for task in self.tasks])

        self.log('training loss', loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        warmup = 16000

        class inverse_square_root_rule():
            def __init__(self, warmup):
                self.warmup = warmup

            def __call__(self, i):
                return min((i+1)/self.warmup, math.sqrt(warmup/i))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, inverse_square_root_rule(warmup))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


# TODO reversibility
# TODO optimize
# TODO dropout
# TODO norm
class AxialTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_heads=None, dim_ff=None, pos_emb=None):
        super().__init__()
        if dim_ff is None:
            dim_ff = 2 * dim
        self.pos_emb = pos_emb
        self.embedding = nn.Sequential(nn.Linear(8, dim), nn.LeakyReLU())

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attn = AxialAttention(dim, num_dimensions=2, heads=heads, dim_heads=None, dim_index=-1, sum_axial_out=True)
            ff = nn.Sequential(nn.Linear(dim, dim_ff), nn.LeakyReLU(), nn.Linear(dim_ff, dim))
            self.layers.append(nn.Sequential(attn, ff))

    def forward(self, x):
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        x = self.embedding(x)
        i = 0
        for layer in self.layers:
            i += 1
            y = layer(x)
            x = x + y
        return x
