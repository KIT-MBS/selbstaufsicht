import inspect
import torch
from torch.nn import Module
from typing import Any, Dict, Tuple


class DifferentiableFunction():
    """Base class for manually differentiable functions."""
    
    @staticmethod
    def forward(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for the forward pass.

        Args:
            ctx (Dict[str, Any]): Computational context.
            *args (Any): Input data and parameters for the differentiable function.
            **kwargs (Any): Keyworded parameters for the differentiable function.

        Raises:
            NotImplementedError: Abstract method.

        Returns:
            Any: Output data
        """
        
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for the backward pass.

        Args:
            ctx (Dict[str, Any]): Computational context.
            *args (Any): Incoming derivatives for the differentiable function.
            **kwargs (Any): Incoming keyworded derivatives for the differentiable function.

        Raises:
            NotImplementedError: Abstract method.

        Returns:
            Any: Derivatives w.r.t. the input data.
        """
        
        raise NotImplementedError

class SoftmaxF(DifferentiableFunction):
    """Implements a manually differentiable version of softmax."""
    
    @staticmethod
    def forward(ctx: Dict[str, Any], x: torch.Tensor, dim: int, training: bool = True) -> torch.Tensor:
        """
        Performs forward pass of the softmax function.

        Args:
            ctx (Dict[str, Any]): Computational context.
            x (torch.Tensor): Input data.
            dim (int): Dimension over which softmax is performed.
            training (bool, optional): Whether training mode is active. Defaults to True.

        Returns:
            torch.Tensor: Softmax output.
        """
        
        assert -x.ndim <= dim < x.ndim
        y = x.softmax(dim=dim)
        ctx['dim'] = dim if dim >= 0 else x.ndim + dim
        ctx['y'] = y
        return y
            
    @staticmethod
    def backward(ctx: Dict[str, Any], grad_y: torch.Tensor) -> torch.Tensor:
        """
        Performs backward pass of the softmax function.

        Args:
            ctx (Dict[str, Any]): Computational context.
            grad_y (torch.Tensor): Incoming derivative w.r.t. the softmax output. 

        Returns:
            torch.Tensor: Derivative w.r.t. the input data.
        """
        
        dim = ctx['dim']
        y = ctx['y']
        ctx['y'] = None
        
        # move selected dim to the end, contract all remaining dims in the preceding dim
        permutation = tuple(range(dim)) + tuple(range(dim + 1, y.ndim)) + (dim,)
        inv_permutation = tuple(range(dim)) + (y.ndim - 1,) + tuple(range(dim, y.ndim - 1))
        dim_size = y.size(dim)
        num_el = y.numel() // dim_size
        
        y = y.permute(*permutation)
        permutation_shape = y.size()
        y = y.reshape(num_el, dim_size)
        
        grad_sm = grad_y.permute(*permutation)
        grad_sm = grad_sm.reshape(num_el, dim_size)
        
        grad_sm = (grad_sm * y)[:, None, :] - (grad_sm[:, None, :] @ y[:, :, None]) @ y[:, None, :]
        grad_sm.squeeze_()
        
        # invert permutation and reshaping
        grad_sm = grad_sm.reshape(*permutation_shape)
        grad_sm = grad_sm.permute(*inv_permutation)
        return grad_sm


class DropoutF(DifferentiableFunction):
    """Implements a manually differentiable version of dropout."""
    
    @staticmethod
    def forward(ctx: Dict[str, Any], x: torch.Tensor, p: float, autocast: bool = False, training: bool = True) -> torch.Tensor:
        """
        Performs forward pass of the dropout function.

        Args:
            ctx (Dict[str, Any]): Computational context.
            x (torch.Tensor): Input data.
            p (float): Dropout probability.
            autocast (bool, optional): Whether autocast is activate. Defaults to False.
            training (bool, optional): Whether training mode is active. Defaults to True.

        Returns:
            torch.Tensor: Dropout output.
        """
        
        assert 0 <= p <= 1
        mask_dtype = torch.float16 if autocast else torch.float32

        if training:
            if p == 0:
                mask = torch.ones_like(x, dtype=mask_dtype, device=x.device)
            elif p == 1:
                mask = torch.zeros_like(x, dtype=mask_dtype, device=x.device)
            else:
                mask = (torch.rand_like(x, dtype=mask_dtype, device=x.device) > p).to(mask_dtype) * (1.0 / (1 - p))
        else:
            mask = torch.ones_like(x, dtype=mask_dtype, device=x.device)
        
        ctx['mask'] = mask
        return mask * x
        
    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> torch.Tensor:
        """
        Performs backward pass of the dropout function.

        Args:
            ctx (Any): Computational context.
            grad_y (torch.Tensor): Incoming derivative w.r.t. the dropout output.

        Returns:
            torch.Tensor: Derivative w.r.t. the input data.
        """
        
        return ctx['mask'] * grad_y


class DifferentiableModule(Module):
    """
    Super class of a pytorch module, which encapsulates a differentiable function and whose backward pass 
    can be invoked manually.
    """
    
    def __init__(self, f: DifferentiableFunction, **params: Any) -> None:
        """
        Initializes the module.

        Args:
            f (DifferentiableFunction): Encapsulated differentiable function.
            **params (Any): Keyworded constant parameters for the differentiable function.  
        """
        
        super().__init__()
        f_signature = inspect.signature(f.forward).parameters
        self._f = f
        self._num_necessary_args = len([arg_name for arg_name, arg in f_signature.items() 
                                        if arg.default is inspect.Parameter.empty and arg_name != 'ctx'])
        self._params = params
        self._ctx = None
    
    def forward(self, *args: Any, no_backward: bool = False, **kwargs: Any) -> Any:
        """
        Performs forward pass of the encapsulated differentiable function.

        Args:
            *args (Any): Input data and parameters for the differentiable function.
            no_backward (bool, optional): Whether computational context should not be cached for a subsequent 
            backward pass. Defaults to False.
            **kwargs (Any): Keyworded parameters for the differentiable function.

        Raises:
            TypeError: The number of passed arguments (including constant parameters) must the greater that or 
            equal to the number of necessary (non-default) arguments of the invoked differentiable function.

        Returns:
            Any: Output of the differentiable function
        """
        
        assert type(self._ctx) is dict
        
        updated_params = {**self._params, **kwargs}
        if len(args) + len(updated_params) < self._num_necessary_args:
            raise TypeError("Number of passed arguments is too small!")
        updated_params['training'] = self.training
        
        # compute forward pass, activate grads and cache the context for the backward pass if intended
        grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(grad_mode and not no_backward)
        out = self._f.forward(self._ctx, *args, **updated_params)
        torch.set_grad_enabled(grad_mode)
        
        return out
    
    def backward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs backward pass of the encapsulated differentiable function.

        Args:
            *args (Any): Incoming derivatives for the backward pass of the differentiable function.
            **kwargs (Any): Keyworded incoming derivatives for the differentiable function.

        Raises:
            TypeError: A computational context from a preceding forward pass is required.
        Returns:
            Any: Derivatives w.r.t. the input data.
        """
        
        assert type(self._ctx) is dict
        
        for cached_data_name, cached_data in self._ctx.items():
            if cached_data is None:
                raise TypeError("Context for \"%s\" not available. You need to run a forward pass beforehand!" % cached_data_name)
        
        # compute derivatives using the cached context and incoming derivatives
        # deactivate grads temporarily, since no second-order derivatives are to be computed
        grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        grads = self._f.backward(self._ctx, *args, **kwargs)
        torch.set_grad_enabled(False)
        
        # empty the context cache
        for cached_data_name, cached_data in self._ctx.items():
            self._ctx[cached_data_name] = None
        
        return grads


class Dropout(DifferentiableModule):
    """Manually differentiable module wrapper for the dropout function."""
    
    def __init__(self, p: int) -> None:
        """
        Initializes the manually differentiable dropout module.

        Args:
            p (int): Dropout probability.
        """
        
        super().__init__(DropoutF, p=p)
        self._ctx = {'mask': None}


class Softmax(DifferentiableModule):
    """Manually differentiable module wrapper for the softmax function."""
    
    def __init__(self) -> None:
        """Initializes the manually differentiable softmax module."""
        
        super().__init__(SoftmaxF)
        self._ctx = {'dim': None, 'y': None}
