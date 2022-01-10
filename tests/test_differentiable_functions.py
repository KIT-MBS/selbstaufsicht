import torch
import torch.testing as testing
import pytest

import selbstaufsicht.modules.differentiable_functions as df


def test_softmax_f():
    x = torch.randn(5, 5)
    
    y, _ = df.SoftmaxF.apply(x, 0)
    y_expected = x.softmax(dim=0)
    testing.assert_close(y, y_expected)
    
    x = torch.randn(5, 5, requires_grad=True)
    y, _ = df.SoftmaxF.apply(x, -1)
    y.sum().backward()
    x_grad = x.grad.clone()
    
    # reset grad
    x.grad.detach_()
    x.grad.zero_()
    
    y_expected = x.softmax(dim=-1)
    y_expected.sum().backward()
    x_grad_expected = x.grad.clone()
    
    testing.assert_close(y, y_expected)
    testing.assert_close(x_grad, x_grad_expected)


def test_dropout_f():
    x = torch.randn(1000, 1000, requires_grad=True)
    
    y, _ = df.DropoutF.apply(x, 0.)
    y_expected = x
    y.sum().backward()
    x_grad = x.grad.clone()
    x_grad_expected = torch.ones_like(x)
    testing.assert_close(y, y_expected)
    testing.assert_close(x_grad, x_grad_expected)
    
    # reset grad
    x.grad.detach_()
    x.grad.zero_()
    
    y, _ = df.DropoutF.apply(x, 1.)
    y_expected = torch.zeros_like(x)
    y.sum().backward()
    x_grad = x.grad.clone()
    x_grad_expected = y_expected
    testing.assert_close(y, y_expected)
    testing.assert_close(x_grad, x_grad_expected)
    
    # reset grad
    x.grad.detach_()
    x.grad.zero_()
    
    ratio_expected = 0.5
    y, _ = df.DropoutF.apply(x, ratio_expected)
    ratio = torch.count_nonzero(y).item() / y.numel()
    y.sum().backward()
    x_grad = x.grad.clone()
    grad_ratio_zero = (x_grad == 0).sum().item() / x_grad.numel()
    grad_ratio_two = (x_grad == 2).sum().item() / x_grad.numel()
    assert grad_ratio_two + grad_ratio_zero == 1
    testing.assert_close(ratio, ratio_expected, atol=1e-3, rtol=1e-2)
    testing.assert_close(grad_ratio_zero, ratio_expected, atol=1e-3, rtol=1e-2)
    testing.assert_close(grad_ratio_two, ratio_expected, atol=1e-3, rtol=1e-2)