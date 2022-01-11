from functools import partial
import inspect
import torch
from torch.nn import Module
from torch.autograd import Function, gradcheck
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Tuple

class SoftmaxF(Function):
    """Implements a manually differentiable version of softmax."""
    
    @staticmethod
    @custom_fwd
    def forward(ctx: Any, x: torch.Tensor, dim: int, training: bool = True) -> Tuple[torch.Tensor, Any]:
        """
        Performs forward pass of the softmax function.

        Args:
            ctx: Computational context.
            x (torch.Tensor): Input data.
            dim (int): Dimension over which softmax is performed.
            training (bool, optional): Whether training mode is active. Defaults to True.

        Returns:
            Tuple[torch.Tensor, Any]: Softmax output; computational context.
        """
        
        assert -x.ndim <= dim < x.ndim
        y = x.softmax(dim=dim)
        ctx.dim = dim if dim >= 0 else x.ndim + dim
        ctx.y = y
        return y, ctx
            
    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad_y: torch.Tensor, grad_ctx: None) -> Tuple[torch.Tensor, None, None]:
        """
        Performs backward pass of the softmax function.

        Args:
            ctx (Any): Computational context.
            grad_y (torch.Tensor): Incoming derivative w.r.t. the softmax output. 
            grad_ctx (None): Dummy argument, required by the autograd engine.

        Returns:
            Tuple[torch.Tensor, None, None]: Derivative w.r.t. the input data; dummy arguments, 
            required by the autograd engine.
        """
        
        # move selected dim to the end, contract all remaining dims in the preceding dim
        permutation = tuple(range(ctx.dim)) + tuple(range(ctx.dim + 1, ctx.y.ndim)) + (ctx.dim,)
        dim_size = ctx.y.size(ctx.dim)
        num_el = ctx.y.numel() // dim_size
        
        y = ctx.y.permute(*permutation)
        permutation_shape = y.size()
        y = y.reshape(num_el, dim_size)
        
        grad_sm = grad_y.permute(*permutation)
        grad_sm = grad_sm.reshape(num_el, dim_size)
        
        grad_sm = (grad_sm * y)[:, None, :] - (grad_sm[:, None, :] @ y[:, :, None]) @ y[:, None, :]
        grad_sm.squeeze_()
        
        # invert permutation and reshaping
        inv_permutation = tuple(range(ctx.dim)) + (ctx.y.ndim - 1,) + tuple(range(ctx.dim, ctx.y.ndim - 1))
        grad_sm = grad_sm.reshape(*permutation_shape)
        grad_sm = grad_sm.permute(*inv_permutation)
        return grad_sm, None, None


class DropoutF(Function):
    """Implements a manually differentiable version of dropout."""
    
    @staticmethod
    @custom_fwd
    def forward(ctx: Any, x: torch.Tensor, p: float, training: bool = True) -> Tuple[torch.Tensor, Any]:
        """
        Performs forward pass of the dropout function.

        Args:
            ctx (Any): Computational context.
            x (torch.Tensor): Input data.
            p (float): Dropout probability.
            training (bool, optional): Whether training mode is active. Defaults to True.

        Returns:
            Tuple[torch.Tensor, Any]: Dropout output; computational context.
        """
        
        assert 0 <= p <= 1
        if training:
            if p == 0:
                mask = torch.ones_like(x, device=x.device)
            elif p == 1:
                mask = torch.zeros_like(x, device=x.device)
            else:
                mask = torch.full_like(x, 1-p, device=x.device).bernoulli() * (1.0 / (1 - p))
        else:
            mask = torch.ones_like(x, device=x.device)
        ctx.mask = mask
        y = mask * x
        return y, ctx
        
    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad_y: torch.Tensor, grad_ctx: None) -> Tuple[torch.Tensor, None, None]:
        """
        Performs backward pass of the dropout function.

        Args:
            ctx (Any): Computational context.
            grad_y (torch.Tensor): Incoming derivative w.r.t. the dropout output. 
            grad_ctx (None): Dummy argument, required by the autograd engine.

        Returns:
            Tuple[torch.Tensor, None, None]: Derivative w.r.t. the input data; dummy arguments, 
            required by the autograd engine.
        """
        
        dr_grad = ctx.mask * grad_y
        return dr_grad, None, None


class DifferentiableModule(Module):
    """
    Super class of a pytorch module, which encapsulates an autograd function and whose backward pass 
    can be invoked manually.
    """
    
    def __init__(self, f: Function, **params: Any) -> None:
        """
        Initializes the module.

        Args:
            f (Function): Encapsulated autograd function.
            params (Dict): Keyworded constant parameters for the autograd function.  
        """
        
        super().__init__()
        f_signature = inspect.signature(f.forward).parameters
        self._f = f
        self._args_dict = {arg_name: arg.default for arg_name, arg in f_signature.items() 
                           if arg_name != 'ctx'}
        self._num_necessary_args = len([arg_name for arg_name, arg_default in self._args_dict.items() 
                                        if arg_default is inspect.Parameter.empty])
        self._params = params
        self._ctx = None
    
    def forward(self, *args: Any, no_backward: bool = False, **kwargs: Any) -> Any:
        """
        Performs forward pass of the encapsulated autograd function.

        Args:
            args (Any): Input data and parameters for the autograd function.
            no_backward (bool, optional): Whether computational context should not be cached for a subsequent 
            backward pass. Defaults to False.
            kwargs (Dict): Keyworded parameters for the autograd function.

        Raises:
            TypeError: The number of passed arguments (including constant parameters) must the greater that or 
            equal to the number of necessary (non-default) arguments of the invoked autograd function.
            TypeError: The autograd function needs to specify at least one differentiable return value

        Returns:
            Any: Output of the autograd function
        """
        
        # merge args and kwargs, s.t. expected argument order is preserved
        updated_params = {**self._params, **kwargs}
        updated_args = []
        arg_idx = 0
        if len(args) + len(updated_params) < self._num_necessary_args:
            raise TypeError("Number of passed arguments is too small!")
        updated_params['training'] = self.training
        for arg_name, arg_default in self._args_dict.items():
            if arg_name in updated_params:
                updated_args.append(updated_params[arg_name])
            else:
                if arg_default is inspect.Parameter.empty:
                    updated_args.append(args[arg_idx])
                    arg_idx += 1
                else:
                    updated_args.append(arg_default)
        
        # compute forward pass, cache the context for the backward pass
        out = self._f.apply(*updated_args)
        self._ctx = None if no_backward else out[-1]
        out = out[:-1] 
        if len(out) > 1:
            return out
        elif len(out) == 1:
            return out[0]
        else:
            raise TypeError("No differentiable return value specified for the forward pass!")
    
    def backward(self, *args: Any) -> Any:
        """
        Performs backward pass of the encapsulated autograd function.

        Args:
            args (Any): Incoming derivatives for the backward pass of the autograd function.

        Raises:
            TypeError: A computational context from a preceding forward pass is required.
            TypeError: The autograd function needs to specify at least one returned derivative.

        Returns:
            Any: Derivatives w.r.t. the input data.
        """
        
        if self._ctx is None:
            raise TypeError("No context available. You need to run a forward pass beforehand!")
        
        # compute derivatives using the cached context and incoming derivatives
        grads = self._f.backward(self._ctx, *args, None)
        # empty the context cache
        self._ctx = None
        out = tuple(grad for grad in grads if grad is not None)
        if len(out) > 1:
            return out
        elif len(out) == 1:
            return out[0]
        else:
            raise TypeError("No return value specified for the backward pass!")


class Dropout(DifferentiableModule):
    """Manually differentiable module wrapper for the dropout function."""
    
    def __init__(self, p: int) -> None:
        """
        Initializes the manually differentiable dropout module.

        Args:
            p (int): Dropout probability.
        """
        
        super().__init__(DropoutF, p=p)


class Softmax(DifferentiableModule):
    """Manually differentiable module wrapper for the softmax function."""
    
    def __init__(self) -> None:
        """Initializes the manually differentiable softmax module."""
        
        super().__init__(SoftmaxF)
