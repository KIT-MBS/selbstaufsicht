import pytorch_lightning as pl
from . import modules
from axial_attention import AxialAttention

class MSAModel(pl.LightningModule):
    """
    Model for pre-training on multiple sequence alignments of biological sequences
    """
    def __init__(self, molecule='RNA'):
        # TODO task parameters
        super().__init__()
        dim = 32

        # self.backbone = models.TransformerEncoderStack(4, 32, 4, 128, 12)
        self.backbone = AxialTransformerEncoder(dim)
        self.demasking_head = models.DemaskingHead(dim, 5)
        self.deshuffling_head = models.DeshufflingHead(dim, 2)

        self.mask_crit = nn.KLDivLoss(reduction='batchmean')
        self.shuf_crit = nn.CrossEntropyLoss()


    def forward(self, encoded_msa_batch):
        """
        Forward pass through the model. Use for inference.
        Args:
            encoded_msa_batch: batch of msas encoded into a tensor of size [batch_size, number_of_sequences, sequence_length, embed_dim]
        """
        latent = self.backbone(encoded_msa)

        return latent

    def shared_step(self, batch_data):
        latent = self(x)
        demasking_result = self.demasking_head(x) # [B, S, L, 5]
        deshuffling_result = self.deshuffling_head(x) # [B]
        return demasking_result, deshuffling_result

    def training_step(self, batch_data, batch_idx):
        # TODO apply task transforms to batch

        result = self.shared_step(batch_data)

        # TODO apply heads
        # TODO compute losses

        return loss

    def validation_step(self, batch_data, batch_idx):
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters())


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
        self.embedding = nn.Sequential([nn.Linear(8, dim), nn.LeakyReLU()])

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = AxialAttention(dim, num_dimensions=2, heads=heads, dim_heads=None, dim_index=-1, sum_axial_out=True)
            ff = nn.Sequential([nn.Linear(dim, dim_ff), nn.LeakyReLU(), nn.Linear(dim_ff, dim)])
            layers.append(nn.Sequential(attn, ff))


    def forward(self, encoded_msa_batch):
        if self.pos_emb is not None:
            encoded_msa_batch = self.pos_emb(encoded_msa_batch)
        for layers in self.layers:
            encoded_msa_batch = encoded_msa_batch + layer(encoded_msa_batch)
        return encoded_msa_batch
