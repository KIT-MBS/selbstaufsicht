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


class EmbeddedJigsawLoss(nn.Module):
    def __init__(self, ignore_value: int = -1, ignore_dim: int = -1, reduction='mean'):
        """
        Initializes L2 loss for the Euclidean embedding of Jigsaw with ignore-index support.

        Args:
            ignore_value (int, optional): Target value which is ignored in comparison. Defaults to -1.
            ignore_dim (int, optional): Target dim which is ignored in mean computation, if it contains only \"ignore_value\". Defaults to -1.
            reduction (str, optional): Reduction mode: mean, sum. Defaults to \"mean\".
        """
        
        super(EmbeddedJigsawLoss, self).__init__()
        self.ignore_value = ignore_value
        self.ignore_dim = ignore_dim
        self.reduction = reduction
        # mean has to be computed manually due to ignore_value
        if reduction == 'mean':
            reduction = 'sum'
        self.fn = nn.MSELoss(reduction=reduction)
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes L2 loss for given predictions and target data.

        Args:
            preds (torch.Tensor): Predictions.
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: L2 loss.
        """
        
        # check if shapes match
        assert preds.shape == target.shape, "Shapes must match!"
        
        # compute and apply ignore mask
        mask = target != self.ignore_value
        loss = self.fn(preds * mask, target * mask)
        
        if self.reduction == 'mean':
            num_total = target.numel()
            # only ignore full vectors with ignore_value
            num_ignore = ((~mask).sum(dim=self.ignore_dim) == mask.shape[self.ignore_dim]).sum() * mask.shape[self.ignore_dim]
            loss /= (num_total - num_ignore)
        
        return loss
