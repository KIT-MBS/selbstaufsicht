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


class BinaryFocalNLLLoss(nn.Module):
    def __init__(self, gamma: float = 2., weight: torch.Tensor = None, ignore_index: int = -1, reduction='mean'):
        """
        Initializes Binary Focal Negative Log Likelihood Loss with ignore-index support.

        Args:
            gamma (float, optional): Exponent of the modulating factor. Defaults to 2..
            weight (torch.Tensor, optional): Weights, by which each class is re-scaled in the loss computation. Defaults to None.
            ignore_index (int, optional): Target value which is ignored in comparison. Defaults to -1.
            reduction (str, optional): Reduction mode: mean, sum. Defaults to \"mean\".
        """
        
        super(BinaryFocalNLLLoss, self).__init__()
        self.gamma = gamma
        if weight is None:
            self.weight = torch.tensor([0.5, 0.5])
        else:
            assert weight.numel() == 2, "Exactly two weights are needed!"
            self.weight = weight
            if self.weight.sum() != 1:
                self.weight /= self.weight.sum()
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes BinaryFocalNLLLoss for given predictions and target data (cf. https://arxiv.org/abs/1708.02002v2).

        Args:
            preds (torch.Tensor): Predictions (one-hot-encoded with class-dim 1).
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: BinaryFocalNLLLoss.
        """
        
        # check if shapes match (exclude class dim for preds due to one-hot encoding)
        assert preds.shape[:1] + preds.shape[2:] == target.shape, "Shapes must match except for class-dim 1!"
        
        # check if there are two classes
        assert preds.shape[1] == 2

        # check if ignore_index is outside of class index range
        assert not 0 <= self.ignore_index < 2, "Parameter \'ignore_index\' must not be in range [0, 2)!"
        
        # compute and apply ignore mask
        mask = target != self.ignore_index
        target_ = target[mask]
        preds_ = preds.permute(0, *(idx for idx in range(2, preds.ndim)), 1)
        preds_ = preds_[mask, :]
        
        # compute focal loss
        loss = torch.zeros(target.shape, device=target.device)
        loss[mask] = -(self.weight[0] * (torch.exp(preds_[:, 1])) ** self.gamma * (1 - target_) * preds_[:, 0] + 
                       self.weight[1] * (torch.exp(preds_[:, 0])) ** self.gamma * target_ * preds_[:, 1])
        
        # perform reduction, if needed
        if self.reduction in ('sum', 'mean'):
            loss = loss.sum()
        if self.reduction == 'mean':
            # divide by sum of applied class weights in order to normalize w.r.t. class weights
            loss /= self.weight[target_].sum()
        
        return loss
    

class DiceNLLLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8, ignore_index: int = -1, reduction='mean'):
        """
        Initializes Dice Negative Log Likelihood Loss with ignore-index support.

        Args:
            epsilon (float, optional): Additive for numerical stabilization. Defaults to 1e-8.
            ignore_index (int, optional): Target value which is ignored in comparison. Defaults to -1.
            reduction (str, optional): Reduction mode: mean, sum. Defaults to \"mean\".
        """
        
        super(DiceNLLLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes DiceNLLLoss for given predictions and target data (cf. https://arxiv.org/abs/1707.03237).

        Args:
            preds (torch.Tensor): Predictions (one-hot-encoded with class-dim 1).
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: DiceNLLLoss.
        """
        
        # check if shapes match (exclude class dim for preds due to one-hot encoding)
        assert preds.shape[:1] + preds.shape[2:] == target.shape, "Shapes must match except for class-dim 1!"
        
        # check if there are two classes
        assert preds.shape[1] == 2

        # check if ignore_index is outside of class index range
        assert not 0 <= self.ignore_index < 2, "Parameter \'ignore_index\' must not be in range [0, 2)!"
        
        # compute and apply ignore mask
        mask = target != self.ignore_index
        target_ = target.unsqueeze(0).repeat(2, *(1 for idx in range(target.ndim)))
        target_[0, mask] = 1 - target_[0, mask]
        target_[0, ~mask] = 0
        target_[1, ~mask] = 0
        # TODO: Maybe add an option to the ContactHead, s.t. it returns Sigmoid output instead of LogSigmoid output
        preds_ = torch.exp(preds)
        preds_ = preds_.permute(1, 0, *(idx for idx in range(2, preds_.ndim))) * mask.unsqueeze(0).expand(2, *(-1 for idx in range(target.ndim)))
        
        # compute dice loss
        image_dims = tuple(idx for idx in range(1, target.ndim))
        loss = (1 - 
                (torch.sum(target_[1, :] * preds_[1, :], dim=image_dims) + self.epsilon) / 
                (torch.sum(target_[1, :] + preds_[1, :], dim=image_dims) + self.epsilon) - 
                (torch.sum(target_[0, :] * preds_[0, :], dim=image_dims) + self.epsilon) / 
                (2 * torch.sum(mask, dim=image_dims) - torch.sum(target_[1, :] - preds_[1, :], dim=image_dims) + self.epsilon))
        
        # perform reduction, if needed
        if self.reduction in ('sum', 'mean'):
            loss = loss.sum()
        if self.reduction == 'mean':
            loss /= target.shape[0]
        
        return loss