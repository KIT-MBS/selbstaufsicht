import math
import torch
import torch.nn as nn

# TODO fix device issues
class SinoidalPositionalEncodingSequence(nn.Module):
    def __init__(self, d, dropout=0.1, max_len=3000):
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
        x = x + self.encoding[:x.size[0], :]
        return self.dropout(x)
