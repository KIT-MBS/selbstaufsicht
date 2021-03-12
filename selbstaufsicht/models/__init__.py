import math
import torch.nn as nn
from ..layers import SinoidalPositionalEncodingSequence

class TransformerEncoderStack(nn.Module):
    def __init__(self, dinput, d, nhead, dff, nlayers, dropout=0.3):
        super(TransformerEncoderStack, self).__init__()
        self.positional_encoding = SinoidalPositionalEncodingSequence(d, dropout)
        self.dinput = dinput
        encoder_layer = TransformerEncoderLayer(d, nhead, dff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, nlayers)
        self.embedding = nn.Embedding(dinput, d)
        self.output_head = nn.Linear(d, dinput)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.dinput)
        src = self.positional_encoding(src)
        src = self.encoder(src)
        src = self.head(src)
        return src
