import pytorch_lightning as pl
from . import modules

class MSAModel(pl.LightningModule):
    """
    Model for pre-training on multiple sequence alignments of biological sequences
    """
    def __init__(self, molecule='RNA'):
        super().__init__()

        self.backbone = models.TransformerEncoderStack(4, 32, 4, 128, 12)
        self.demasking_head = models.DemaskingHead(32, 4)
        self.deshuffling_head = models.DeshufflingHead(32, 2)
        # TODO crit per task
        self.crit = nn.KLDivLoss(reduction='batchmean')

    def forward(self, encoded_msa_batch):
        """
        Forward pass through the model. Use for inference.
        Args:
            encoded_msa_batch: batch of msas encoded into a tensor of size [batch_size, number_of_sequences, sequence_length, embed_dim]
        """
        latent = self.backbone(encoded_msa)

        return latent

    # TODO figure out actual input dim ordering
    def shared_step(self, batch_data, batch_idx):
        # TODO prep input
        latent = self(x)
        demasking_result = self.demasking_head(x)
        deshuffling_result = self.deshuffling_head(x)
        return demasking_result, deshuffling_result

    def training_step(self, batch_data, batch_idx):
        result = self.shared_step(batch_data, batch_idx)
        batch_input, target = batch_data # [input] == [B, N, S, D]

        loss = self.crit(result, target)
        pred = self(x)
        pred = nn.functional.log_softmax(x, dim=-1)
        loss = self.crit(y, pred)
        return loss

    def validation_step(self, batch_data, batch_idx):
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters())
