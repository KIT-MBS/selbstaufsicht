import math
import torch
from torch import nn

import pytorch_lightning as pl
from axial_attention import AxialAttention

# from . import modules
from selbstaufsicht.modules import Transmorpher

# NOTE for using simCLR loss from bolts
# from pytorch_lightning.models.self_supervised.simclr.simclr_module import SyncFunction


class MSAModel(pl.LightningModule):
    """
    Model for pre-training on multiple sequence alignments of biological sequences
    """
    def __init__(
            self,
            num_layers=4,
            num_heads=4,
            dim=32,
            permutations=None,
            heads=None,
            ):
        # TODO task and model parameters
        super().__init__()

        self.backbone = Transmorpher()
        self.tasks = [t for t in heads.keys()]
        self.heads = heads
        # self.demasking_head = modules.DemaskingHead(dim, 5)
        # self.deshuffling_head = modules.DeshufflingHead(dim, n_permutations)
        # self.contrastive_head = modules.ContrastiveHead()

        self.demasking_loss = nn.KLDivLoss(reduction='batchmean')
        self.deshuffling_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = nn.CosineEmbeddingLoss()

    def forward(self, encoded_msa_batch):
        """
        Forward pass through the model. Use for inference.
        Args:
            encoded_msa_batch: batch of msas encoded into a tensor of size [batch_size, number_of_sequences, sequence_length, embed_dim]
        """
        latent = self.backbone(encoded_msa_batch)
        return latent

    def training_step(self, batch_data, batch_idx):
        # TODO move as much as possible into the collator
        batch_input, lens = batch_data
        batch_size = batch_input.size(0)
        batch_input = batch_input[:, torch.randperm(batch_input.size(1)), :, :]

        mask_start = torch.randint(batch_input.size(2) - self.mask_width, size=(1,))

        in1_sequences_end = min(self.n_sequences, batch_input.size(1) // 2)
        sequences_overlap = in1_sequences_end // 10
        # TODO different/random sizes of partitions
        in2_sequences_start = in1_sequences_end - sequences_overlap
        in2_sequences_end = max(batch_input.size(1), in2_sequences_start + in1_sequences_end // 2)
        input1 = batch_input[:, :in1_sequences_end]
        original = input1.clone().detach()
        input2 = batch_input[:, in2_sequences_start: in2_sequences_end]

        input1[:, :, mask_start:mask_start + self.mask_width, 0:6] = torch.tensor([0., 0., 0., 0., 0., 1.])

        demasking_target = original[:, :, mask_start:mask_start + self.mask_width, 0:5]
        # deshuffling_target = torch.randint(self.n_partitions, batch_input.size(0))
        # input1, deshuffling_target = jigsaw(input1, self.shuffle_partitions)

        latent = self(input1)

        demasking_output = self.demasking_head(latent)[:, :, mask_start:mask_start + self.mask_width, 0:5]
        # deshuffling_output = self.deshuffling_head(latent)

        demasking_loss = self.demasking_loss(demasking_output, demasking_target)
        # deshuffling_loss = self.deshuffling_loss(deshuffling_output, deshuffling_target)
        # TODO should the contrastive target/input also be masked/shuffled?
        contrastive_loss = self.contrastive_loss(latent.sum(dim=1).reshape(batch_size, -1), self(input2).sum(dim=1).reshape(batch_size, -1), torch.ones(batch_input.size(0)))

        # loss = demasking_loss + deshuffling_loss + contrastive_loss
        loss = demasking_loss + contrastive_loss
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


# TODO better compartmentalization of tasks
# TODO optimize
def jigsaw(unshuffled_input, permutations=None):
    """
    unshuffled_input: pre processed input to jigsaw [B, S, L, D]
    deshuffling_target: permutation to apply to jigsaw pieces [B, number of possible permutations]
    """
    if permutations is None:
        permutations = [[1, 0], ]

    n_partitions = len(permutations[0])
    shuffled_input = torch.zeros_like(unshuffled_input)
    deshuffling_target = torch.randint(n_partitions, unshuffled_input.size(0))
    for i in range(deshuffling_target.size(0)):
        raise
    return shuffled_input, deshuffling_target


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
