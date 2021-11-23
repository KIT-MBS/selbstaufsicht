import math
import torch
from torch import nn

import pytorch_lightning as pl

from selbstaufsicht.modules import Transmorpher2d, TransmorpherLayer2d

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
            aux_input_dim=2,
            attention='tied',
            activation='relu',
            layer_norm_eps=1e-5,
            in_dict_size=7,
            lr=1e-4,
            lr_warmup=16000,
            padding_token=None,
            task_heads=None,
            task_losses=None,
            metrics=None,
            need_attn=False,
            device=None,
            dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        d = num_heads * dim_head

        assert d - aux_input_dim > 0
        self.embedding = nn.Embedding(in_dict_size, d - aux_input_dim, padding_idx=padding_token)
        block = TransmorpherLayer2d(dim_head, num_heads, 2 * dim_head * num_heads, attention=attention, activation=activation, layer_norm_eps=layer_norm_eps, **factory_kwargs)
        self.backbone = Transmorpher2d(block, num_layers, nn.LayerNorm(d, eps=layer_norm_eps, **factory_kwargs))
        if task_heads is not None:
            self.tasks = [t for t in task_heads.keys()]
        self.task_heads = task_heads
        self.losses = task_losses
        self.metrics = metrics
        if task_heads is not None:
            assert self.task_heads.keys() == self.losses.keys()
        self.lr = lr
        self.lr_warmup = lr_warmup
        if need_attn:
            raise NotImplementedError('Extracting attention maps not yet implemented')
        self.need_attn = need_attn

    def forward(self, x, padding_mask=None, aux_features=None):
        """
        Forward pass through the model. Use for inference.
        Args:
            batch is a tuple of an input dict and a target dict
            the input dict contains a tokenized msa, any auxiliary features ('aux_features')
            and any additional for the task heads e.g. the mask where the loss is to be measured for the inpainting task.
            the output dict contains the target per task loss keyed per task
        """

        # NOTE feature dim = -1
        # TODO optimize embedding
        x = self.embedding(x)
        if aux_features is not None:
            aux_features = aux_features.expand(-1, x.size(1), -1, -1)
            x = torch.cat((x, aux_features), dim=-1)
        # TODO extract attention maps
        latent = self.backbone(x, padding_mask, self.need_attn)
        return latent

    def training_step(self, batch_data, batch_idx):
        x, y = batch_data

        latent = self(x['msa'], x.get('padding_mask', None), self.need_attn, x.get('aux_features', None))
        if 'contrastive' in self.tasks:
            if x['msa'].size(0) == 1:
                print('WARN: contrastive task is not really going to work with batch_size==1')
            y['contrastive'] = self.task_heads['contrastive'](self(x['contrastive'], x.get('aux_features', None)), x)

        preds = {task: self.task_heads[task](latent, x) for task in self.tasks}
        lossvals = {task: self.losses[task](preds[task], y[task]) for task in self.tasks}
        for task in self.tasks:
            for m in self.metrics[task]:
                self.log(f'{task} {m}: ', self.metrics[task][m](preds[task], y[task]))
        # TODO weights
        loss = sum([lossvals[task] for task in self.tasks])
        for task in self.tasks:
            self.log(f'{task} loss', lossvals[task])

        self.log('training loss', loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        class inverse_square_root_rule():
            def __init__(self, warmup):
                self.warmup = warmup

            def __call__(self, i):
                return min((i + 1) / self.warmup, math.sqrt(self.warmup / (i + 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, inverse_square_root_rule(self.lr_warmup))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
