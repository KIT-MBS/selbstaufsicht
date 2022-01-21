import torch
from torch import nn

class Accuracy(nn.Module):
    def __init__(self, class_dim: int = -1, ignore_index: int = -1, preds_one_hot: bool = True):
        """
        Initializes accuracy metric with ignore-index support and specifiable class dimension.

        Args:
            class_dim (int, optional): Class dimension, which is used for comparison. All other dimensions are treated as batch dimensions. Defaults to -1.
            ignore_index (int, optional): Class index which is ignored in comparison. Defaults to -1.
            preds_one_hot (bool, optional): Whether \"preds\" is one-hot-encoded in \"class_dim\". Defaults to True.
        """
        
        super(Accuracy, self).__init__()
        self.class_dim = class_dim
        self.ignore_index = ignore_index
        self.preds_one_hot = preds_one_hot
    
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes accuracy for given predictions and target data.

        Args:
            preds (torch.Tensor): Predictions.
            target (torch.Tensor): Target data. Is expected to not contain class dim as \"preds\", but class indices.

        Returns:
            torch.Tensor: Accuracy.
        """
        
        if self.preds_one_hot:
            # check if shapes match (exclude class dim for preds due to one-hot encoding)
            assert preds.shape[:self.class_dim] + (preds.shape[self.class_dim + 1:] if self.class_dim != -1 else ()) == target.shape, "Shapes must match except for \'class_dim\'!"
        else:
            assert preds.shape == target.shape, "Shapes must match!"
        
        # check if ignore_index is outside of class index range
        num_classes = preds.shape[self.class_dim]
        assert not 0 <= self.ignore_index < num_classes, "Parameter \'ignore_index\' must not be in range [0, num_classes)!"
        
        num_total = target.numel()
        num_ignore = target[target == self.ignore_index].numel()
        
        if self.preds_one_hot:
            preds_argmax = torch.argmax(preds, dim=self.class_dim)
        else:
            preds_argmax = preds
        num_correct = (preds_argmax == target).sum()
        
        return num_correct / (num_total - num_ignore)
