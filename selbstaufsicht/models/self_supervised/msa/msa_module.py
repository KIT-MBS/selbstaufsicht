import torch
from torch import nn

import pytorch_lightning as pl
from axial_attention import AxialAttention

from . import modules

# TODO do positional encoding after task preprocessing
class MSAModel(pl.LightningModule):
    """
    Model for pre-training on multiple sequence alignments of biological sequences
    """
    def __init__(self, molecule='RNA', mask_width=1, shuffle_partitions=2, depth=4, heads=4, dim=32):
        # TODO task parameters
        # TODO model parameters
        super().__init__()

        if molecule != 'RNA':
            raise NotImplementedError()

        self.mask_width = mask_width

        # self.backbone = modules.TransformerEncoderStack(4, 32, 4, 128, 12)
        self.backbone = AxialTransformerEncoder(dim, depth=depth, heads=heads)
        self.demasking_head = modules.DemaskingHead(dim, 5)
        deshuffling_head = modules.DeshufflingHead(dim, 2)
        # contrastive_head = modules.ContrastiveHead()
        # self.heads = [demasking_head, deshuffling_head, contrastive_head]

        # self.criteria = [nn.KLDivLoss(reduction='batchmean'), nn.CrossEntropyLoss()]
        self.criteria = [nn.KLDivLoss(reduction='batchmean')]


    def forward(self, encoded_msa_batch):
        """
        Forward pass through the model. Use for inference.
        Args:
            encoded_msa_batch: batch of msas encoded into a tensor of size [batch_size, number_of_sequences, sequence_length, embed_dim]
        """
        latent = self.backbone(encoded_msa_batch)

        return latent

    # TODO
    # def shared_step(self, batch_data):
    #     latent = self(x)
    #     demasking_result = self.demasking_head(x) # [B, S, L, 5]
    #     deshuffling_result = self.deshuffling_head(x) # [B]
    #     return demasking_result, deshuffling_result

    def training_step(self, batch_data, batch_idx):
        batch_input, lens = batch_data
        # batch_input, targets = self.task_transforms(batch_input)
        mask_start = torch.randint(batch_input.size(2)-self.mask_width, size=(1,))
        # masked_input, inpainting_mask = mask(batch_input)
        original = torch.tensor(batch_input)
        batch_input[:,:,mask_start:mask_start+self.mask_width,0:6] = torch.tensor([0.,0.,0.,0.,0.,1.])
        demasking_target = original[:,:,mask_start:mask_start+self.mask_width,0:5]

        latent = self(batch_input)

        demasking_output = self.demasking_head(latent)[:,:,mask_start:mask_start+self.mask_width,0:5]
        # loss = self.criteria[0](outputs, targets)
        loss = self.criteria[0](demasking_output, demasking_target)
        self.log('loss', loss)
        return loss

    # def heads(self, latent, inpainting_mask):
    #     return (self.demasking_head(latent, inpainting_mask), self.deshuffling_head(latent))

    # def compute_head_output(self, latent):
    #     return [h(latent) for h in self.heads]

    # def compute_head_loss(self, outputs, targets):
    #     # TODO log component losses
    #     trips = zip(outputs, targets, self.criteria)
    #     return sum([l(o, t) for (o, t, l) in trips])

    # def task_transforms(self, batch_input):
    #     # TODO modularize
    #     demasking_target = torch.Tensor(batch_input)
    #     masked_input, inpainting_mask = mask(batch_input)
    #     demasking_target = demasking_target[:, :, demasking_target[0]:demasking_target[0]+demasking_target[1]]
    #     assert demasking_target.size() == (batch_input.size(0), batch_input.size(1), self.mask_width, 8)
    #     return masked_input, (demasking_target), (inpainting_mask)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

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
            dim_ff = 2*dim
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
