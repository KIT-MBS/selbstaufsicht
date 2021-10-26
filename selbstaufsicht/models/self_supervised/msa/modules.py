# import math
import torch.nn as nn


class InpaintingHead(nn.Module):
    # TODO channel last
    def __init__(self, d, num_classes, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(InpaintingHead, self).__init__()
        # TODO assuming the last layer in an encoder block is a nonlinearity
        self.num_classes = num_classes
        self.proj = nn.Linear(d, num_classes, **factory_kwargs)

    # TODO the output is basically flattened (of  shape (-1, num_classes)) since the number of masked tokens per sample in the batch is not the same
    def forward(self, latent, x):
        # bs, S, L, D = latent.size()
        output = self.proj(latent)
        # output = output.permute(0, 2, 3, 1)
        # output = output[x['mask'].unsqueeze(-1).expand(-1, -1, -1, self.num_classes)]
        output = output[x['mask']]

        output = output.reshape(-1, self.num_classes)
        return output


class JigsawHead(nn.Module):
    def __init__(self, d, nclasses, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(JigsawHead, self).__init__()
        self.proj = nn.Linear(d, nclasses, **factory_kwargs)

    def forward(self, latent, x):
        # x is of shape [b, c, s, l]
        latent = latent.mean(dim=-1)
        latent = latent.mean(dim=-1)
        return self.proj(latent)


# TODO
# class MaximizingMutualInformationHead():
#     def __init__(self, x):
#         raise NotImplementedError()
