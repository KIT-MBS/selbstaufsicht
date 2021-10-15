import torch
import torch.testing as testing

from selbstaufsicht.modules.layernorm import AxialLayerNorm


def test_layernorm_simple():
    tmp = (torch.arange(10)*10.0).reshape((5, 2))
    module = AxialLayerNorm(1, 2, elementwise_affine=False)
    module2 = torch.nn.LayerNorm((2,))
    tmp2 = module(tmp)
    tmp3 = module2(tmp)
    testing.assert_allclose(tmp2, tmp3)

def test_layernorm_nonstandardaxis():
    tmp = (torch.arange(10, dtype=torch.double)*10.0).reshape((5, 2))
    module = AxialLayerNorm(0, 5, elementwise_affine=False)
    tmp2 = module(tmp)
    testing.assert_allclose(tmp2[:, 0], tmp2[:, 1])

def test_layernorm_mean():
    tmp = (torch.arange(10, dtype=torch.double)*10.0).reshape((5, 2))
    module = AxialLayerNorm(0, 5, elementwise_affine=False)
    tmp2 = torch.sum(module(tmp), dim=0)
    testing.assert_allclose(tmp2, torch.zeros_like(tmp2))

def test_layernorm_stddev():
    tmp = (torch.arange(10, dtype=torch.double)*10.0).reshape((5, 2))
    module = AxialLayerNorm(0, 5, elementwise_affine=False)
    tmp2 = torch.sqrt(torch.sum(module(tmp)**2, dim=0)/tmp.shape[0])
    testing.assert_allclose(tmp2, torch.ones_like(tmp2))
