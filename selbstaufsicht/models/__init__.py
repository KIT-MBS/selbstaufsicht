import math
import torch.nn as nn
from ..layers import SinoidalPositionalEncodingSequence

class TransformerEncoderStack(nn.Module):
    def __init__(self, dinput, d, nhead, dff, nlayers, dropout=0.3):
        super(TransformerEncoderStack, self).__init__()
        self.positional_encoding = SinoidalPositionalEncodingSequence(d, dropout)
        self.dinput = dinput
        encoder_layer = nn.TransformerEncoderLayer(d, nhead, dff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.embedding = nn.Embedding(dinput, d)
        self.output_head = nn.Linear(d, dinput)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.dinput)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.head(x)
        return x
