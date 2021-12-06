import torch
from torch import nn
from pl_bolts.losses.self_supervised_learning import nt_xent_loss


class NT_Xent_Loss(nn.Module):
    def __init__(self, temperature: float) -> None:
        """
        Initializes NT Xent Loss (SimCLR).

        Args:
            temperature (float): Distillation temperature.
        """
        
        super(NT_Xent_Loss, self).__init__()
        self.temperature = temperature

    # TODO ask about the distributed stuff, have to reimplement with our own solution?
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes NT Xent Loss.

        Args:
            x1 (torch.Tensor): First input data.
            x2 (torch.Tensor): Second input data.

        Returns:
            torch.Tensor: Loss value.
        """
        
        return nt_xent_loss(x1, x2, self.temperature)
