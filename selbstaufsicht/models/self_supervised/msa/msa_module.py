import torch
from torch import nn

import pytorch_lightning as pl
from axial_attention import AxialAttention

from . import modules


class MSAModel(pl.LightningModule):
    """
    Model for pre-training on multiple sequence alignments of biological sequences
    """
    def __init__(
            self,
            molecule='RNA',
            mask_width=1,
            shuffle_partitions=2,
            depth=4,
            heads=4,
            dim=32,
            n_sequences=10000):
        # TODO task and model parameters
        super().__init__()

        if molecule != 'RNA':
            raise NotImplementedError()

        self.mask_width = mask_width
        self.n_sequences = n_sequences

        # TODO replace axial transformer with self built optimized module, to get rid of all the unncecessary permutations
        self.backbone = AxialTransformerEncoder(dim, depth=depth, heads=heads)
        self.demasking_head = modules.DemaskingHead(dim, 5)
        self.deshuffling_head = modules.DeshufflingHead(dim, 2)
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
        batch_input = batch_input[:, torch.randperm(batch_input.size(1)), :, :]

        mask_start = torch.randint(batch_input.size(2) - self.mask_width, size=(1,))
        original = batch_input.clone().detach()

        n_sequences = min(self.n_sequences, batch_input.size(1) // 2)
        sequences_overlap = n_sequences // 10
        in2_sequences_start = n_sequences - sequences_overlap
        in2_sequences_end = min(batch_input.size(1), in2_sequences_start + n_sequences // 2)
        input1 = batch_input[:, :n_sequences]
        input2 = batch_input[:, in2_sequences_start, in2_sequences_end]

        input1[:, :, mask_start:mask_start + self.mask_width, 0:6] = torch.tensor([0., 0., 0., 0., 0., 1.])

        demasking_target = original[:, :, mask_start:mask_start + self.mask_width, 0:5]
        deshuffling_target = torch.randint(self.n_partitions, batch_input.size(0))
        input1 = jigsaw(input1, deshuffling_target)

        latent = self(input1)

        demasking_output = self.demasking_head(latent)[:, :, mask_start:mask_start + self.mask_width, 0:5]
        deshuffling_output = self.deshuffling_head(latent)

        demasking_loss = self.demasking_loss(demasking_output, demasking_target)
        deshuffling_loss = self.deshuffling_loss(deshuffling_output, deshuffling_target)
        contrastive_loss = self.contrastive_loss(latent.sum(dim=1), self(input2).sum(dim=1))

        loss = demasking_loss + deshuffling_loss + contrastive_loss
        loss = self.criteria[0](demasking_output, demasking_target)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def jigsaw(deshuffling_input):
    return

# def mask(batch_data):
#     width = self.mask_width
#     start = torch.randint(batch_data.size(2)-width)
#     batch_data[:,:,start:start+width,0:6] = torch.tensor([0.,0.,0.,0.,0.,1.])
#     # inpainting_mask = torch.zeros(batch_data.size()[:-1])
#     # inpainting_mask[:,:,start:start+width] = 1
#     inpainting_mask = (start, width)
#     return batch_data, inpainting_mask


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

    def forward(self, encoded_msa_batch):
        if self.pos_emb is not None:
            encoded_msa_batch = self.pos_emb(encoded_msa_batch)
        encoded_msa_batch = self.embedding(encoded_msa_batch)
        for layer in self.layers:
            encoded_msa_batch = encoded_msa_batch + layer(encoded_msa_batch)
        return encoded_msa_batch
