import torch
from selbstaufsicht.modules.layernorm import *
import unittest

class TestAxialLayerNorm(unittest.TestCase):
    def test_layernorm_simple(self):
        tmp = (torch.arange(10)*10.0).reshape((5,2))
        module = AxialLayerNorm(1, 2, elementwise_affine=False)
        module2 = torch.nn.LayerNorm((2,))
        tmp2 = module(tmp)
        tmp3 = module2(tmp)
        self.assertTrue(torch.isclose(tmp2,tmp3).all())

    def test_layernorm_nonstandardaxis(self):
        tmp = (torch.arange(10,dtype=torch.double)*10.0).reshape((5,2))
        module = AxialLayerNorm(0, 5, elementwise_affine=False)
        tmp2 = module(tmp)
        self.assertTrue(torch.isclose(tmp2[:,0],tmp2[:,1]).all())

    def test_layernorm_mean(self):
        tmp = (torch.arange(10,dtype=torch.double)*10.0).reshape((5,2))
        module = AxialLayerNorm(0, 5, elementwise_affine=False)
        tmp2 = torch.sum(module(tmp),dim=0)
        self.assertTrue(torch.isclose(tmp2,torch.zeros_like(tmp2)).all())

    def test_layernorm_stddev(self):
        tmp = (torch.arange(10,dtype=torch.double)*10.0).reshape((5,2))
        module = AxialLayerNorm(0, 5, elementwise_affine=False)
        tmp2 = torch.sqrt(torch.sum(module(tmp)**2,dim=0)/tmp.shape[0])
        self.assertTrue(torch.isclose(tmp2,torch.ones_like(tmp2)).all())
