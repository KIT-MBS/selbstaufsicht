import math
import torch.nn as nn


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


# TODO
class MaximizingMutualInformationHead(self, x):
    raise NotImplementedError()
