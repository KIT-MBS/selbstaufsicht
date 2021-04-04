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

# ambuguous RNA letters: GAUCRYWSMKHBVDN
rna_to_tensor_dict = {
        '-': torch.tensor([0., 0., 0., 0.]),
        '.': torch.tensor([0., 0., 0., 0.]),
        'G': torch.tensor([1., 0., 0., 0.]),
        'A': torch.tensor([0., 1., 0., 0.]),
        'U': torch.tensor([0., 0., 1., 0.]),
        'C': torch.tensor([0., 0., 0., 1.]),
        'R': torch.tensor([1., 1., 0., 0.])/2.,
        'Y': torch.tensor([0., 0., 1., 1.])/2.,
        'W': torch.tensor([0., 1., 1., 0.])/2.,
        'S': torch.tensor([1., 0., 0., 1.])/2.,
        'M': torch.tensor([0., 1., 0., 1.])/2.,
        'K': torch.tensor([1., 0., 1., 0.])/2.,
        'H': torch.tensor([0., 1., 1., 1.])/3.,
        'B': torch.tensor([1., 0., 1., 1.])/3.,
        'V': torch.tensor([1., 1., 0., 1.])/3.,
        'D': torch.tensor([1., 1., 1., 0.])/3.,
        'N': torch.tensor([1., 1., 1., 1.])/4.,
        }


class EncoderStackTransforms():
    def __init__(self):
        pass

    def __call__(self, msa):
        S = len(msa)
        L = msa.get_alignment_length()
        A = 4
        x = torch.empty()
