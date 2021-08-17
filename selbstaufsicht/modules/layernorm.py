import torch


class AxialLayerNorm(torch.nn.Module):
    def __init__(self, axis, axis_length, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AxialLayerNorm, self).__init__()

        self.axis = axis
        self.axis_length = axis_length
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty((self.axis_length,), **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty((self.axis_length,), **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        tmp = torch.swapaxes(input, self.axis, -1)
        tmp2 = torch.nn.functional.layer_norm(tmp, (self.axis_length,), self.weight, self.bias, self.eps)
        return torch.swapaxes(tmp2, self.axis, -1)
