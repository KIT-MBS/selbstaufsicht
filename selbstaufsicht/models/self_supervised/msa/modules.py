import math
import torch.nn as nn
from ..layers import SinoidalPositionalEncodingSequence

class SetOfSequencesBackbone(nn.Module):
    def __init__(self, dinput, d, nhead, dff, nlayers, dropout=0.3):
        super(SetOfSequencesBackbone, self).__init__()
        self.positional_encoding = SinoidalPositionalEncodingSequence(d, dropout)
        self.dinput = dinput
        self.embedding = nn.Linear(dinput, d)
        encoder_layer = nn.TransformerEncoderLayer(d, nhead, dff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.dinput)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return x

# TODO reduce evo dimension mode
class DemaskingHead(nn.Module):
    def __init__(self, d, doutput):
        super(DemaskingHead, self).__init__()
        self.output_head = nn.Linear(d, doutput)

    def forward(self, x):
        return self.output_head(x)

# TODO
class DeshufflingHead(nn.Module):
    def __init__(self, d, nclasses):
        super(DeshufflingHead, self).__init__()
        self.output_head = nn.Linear(d, nclasses)

    def forward(self, x):
        raise NotImplementedError()
        x = x.sum(dim=1)
        return

# TODO fix device issues
class SinoidalPositionalEncodingSequence(nn.Module):
    def __init__(self, d, dropout=0.1, max_len=24000):
        super(SinoidalPositionalEncodingSequence, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        encoding_buffer = torch.zeros(max_len, d) # max_len, d
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # max_len, 1
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) /d )) # d/2
        encoding_buffer[:, 0::2] = torch.sin(positions * div_term)
        encoding_buffer[:, 1::2] = torch.cos(positions * div_term) # max_len, d
        encoding_buffer = encoding_buffer.unsqueeze(1) # max_len, 1, d

        self.register_buffer('encoding', encoding_buffer)

    def forward(self, x):
        """
        input shape:  sequence length, batch size, embed dim
        """

        x = x + self.encoding[:x.size(0), :]
        return self.dropout(x)
