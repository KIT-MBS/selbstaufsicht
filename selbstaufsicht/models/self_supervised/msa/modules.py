# import math
import torch.nn as nn


class InpaintingHead(nn.Module):
    def __init__(self, d, doutput):
        super(InpaintingHead, self).__init__()
        # TODO assuming the last layer in an encoder block is a nonlinearity
        self.output_head = nn.Linear(d, doutput)

    def forward(self, x):
        return self.output_head(x)


# TODO
# class JigsawHead(nn.Module):
#     def __init__(self, d, nclasses):
#         super(JigsawHead, self).__init__()
#         self.output_head = nn.Linear(d, nclasses)
#
#     def forward(self, x):
#         raise NotImplementedError()
#         x = x.sum(dim=1)


# TODO
# class MaximizingMutualInformationHead():
#     def __init__(self, x):
#         raise NotImplementedError()
