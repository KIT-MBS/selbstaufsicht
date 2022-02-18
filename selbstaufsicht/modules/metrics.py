import torch
from torch import nn

from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, class_dim: int = -1, ignore_index: int = -1, preds_one_hot: bool = True, dist_sync_on_step=False) -> None:
        """
        Initializes accuracy metric with ignore-index support and specifiable class dimension.

        Args:
            class_dim (int, optional): Class dimension, which is used for comparison. All other dimensions are treated as batch dimensions. Defaults to -1.
            ignore_index (int, optional): Class index which is ignored in comparison. Defaults to -1.
            preds_one_hot (bool, optional): Whether \"preds\" is one-hot-encoded in \"class_dim\". Defaults to True.
            dist_sync_on_step (bool, optional): Synchronize metric state across processes at each forward() before returning the value at the step. Defaults to False.
        """

        super(Accuracy, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_dim = class_dim
        self.ignore_index = ignore_index
        self.preds_one_hot = preds_one_hot
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ignore", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates internal metric states by evaluating given predictions and target data.

        Args:
            preds (torch.Tensor): Predictions.
            target (torch.Tensor): Target data. If \"preds_one_hot\", it is expected to not contain class dim as \"preds\", but class indices.
        """
        
        if self.preds_one_hot:
            # check if shapes match (exclude class dim for preds due to one-hot encoding)
            assert preds.shape[:self.class_dim] + (preds.shape[self.class_dim + 1:] if self.class_dim != -1 else ()) == target.shape, "Shapes must match except for \'class_dim\'!"

            # check if ignore_index is outside of class index range
            num_classes = preds.shape[self.class_dim]
            assert not 0 <= self.ignore_index < num_classes, "Parameter \'ignore_index\' must not be in range [0, num_classes)!"
        else:
            assert preds.shape == target.shape, "Shapes must match!"

        num_total = target.numel()
        num_ignore = target[target == self.ignore_index].numel()

        if self.preds_one_hot:
            preds_argmax = torch.argmax(preds, dim=self.class_dim)
        else:
            preds_argmax = preds
        num_correct = (preds_argmax == target).sum()
        
        self.correct += num_correct
        self.total += num_total
        self.ignore += num_ignore
    
    def compute(self) -> torch.Tensor:
        """
        Computes accuracy for accumulated internal metric states.

        Returns:
            torch.Tensor: Accuracy.
        """
        
        return self.correct.float() / (self.total - self.ignore)


class EmbeddedJigsawAccuracy(Accuracy):
    def __init__(self, euclid_emb: torch.Tensor, ignore_value: int = -1) -> None:
        """
        Initializes accuracy metric for the Euclidean embedding of the jigsaw task, including ignore-value support and specifiable class dimension.

        Args:
            euclid_emb (torch.Tensor): Euclidean embedding of the discrete permutation metric.
            ignore_value (int, optional): Target value which is ignored in comparison. Defaults to -1.
        """

        super(EmbeddedJigsawAccuracy, self).__init__(ignore_index=ignore_value, preds_one_hot=False)
        self.euclid_emb = euclid_emb
        self.euclid_emb_device_flag = False
        self.ignore_value = ignore_value

    def permutation_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compares each M-dimensional vector of x with each row vector of the Euclidean embedding.
        Yields the row index with the lowest L2 distance for each M-dimensional vector of x.

        Args:
            x (torch.Tensor): Data w.r.t. the Euclidean embedding [*, M].

        Returns:
            torch.Tensor: Permutation indices.
        """

        assert x.shape[self.class_dim] == self.euclid_emb.shape[1]

        # expand x by permutation dim, permute permutation dim and embedding dim
        temp = x.unsqueeze(-1).expand(*x.shape, self.euclid_emb.shape[0])  # [*, M, P]
        temp = temp.transpose(-1, -2)  # [*, P, M]

        # expand embedding to shape of x
        euclid_emb = self.euclid_emb.view(*((1,) * (x.dim() - 1) + self.euclid_emb.shape)).expand(*(x.shape[:-1] + self.euclid_emb.shape))  # [*, P, M]

        # compute L2 distances over class_dim
        temp = torch.linalg.vector_norm(temp - euclid_emb, dim=-1)  # [*, P]

        # find indices of minimum L2 distances over permutation dim
        return torch.argmin(temp, dim=-1)  # [*]

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates internal metric states by evaluating given predictions and target data by taking the closest permutation index from the Euclidean embedding 
        for predictions, according to the L2 norm.

        Args:
            preds (torch.Tensor): Predictions.
            target (torch.Tensor): Target data.
        """
        
        assert preds.shape == target.shape, "Shapes must match!"

        # necessary for pytorch lightning to push the tensor onto the correct cuda device
        if not self.euclid_emb_device_flag:
            self.euclid_emb = self.euclid_emb.type_as(preds)
            self.euclid_emb_device_flag = True

        # invert embedding: get permutation indices
        preds_indices = self.permutation_indices(preds)
        target_indices = self.permutation_indices(target)

        # compute ignore mask
        mask = (target == self.ignore_value).sum(dim=-1) == target.shape[-1]

        # apply ignore_mask only to targets
        target_indices[mask] = self.ignore_index

        super().update(preds_indices, target_indices)


class BinaryTopLPrecision(Metric):
    def __init__(self, ignore_idx: int = -1, diag_shift: int = 4, dist_sync_on_step: bool = False) -> None:
        """
        Initializes binary top-L precision metric with ignore index support.

        Args:
            ignore_idx (int, optional): Ignored index in the target values. Defaults to -1.
            diag_shift (int, optional): Diagonal offset for predictions. Defaults to 4.
            dist_sync_on_step (bool, optional): Synchronize metric state across processes at each forward() before returning the value at the step. Defaults to False.
        """
        
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_idx = ignore_idx
        self.diag_shift = diag_shift
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates internal metric states by comparing predicted and target values.

        Args:
            preds (torch.Tensor): Predictions [B, 2, L, L].
            target (torch.Tensor): Targets [B, L, L].
        """
        
        assert preds.shape[1] == 2
        B, L, _ = target.shape
        preds_ = preds[:, 1, :, :].squeeze(1)  # [B, L, L]
        assert preds_.size() == target.size()
        
        preds_ = torch.triu(preds_, self.diag_shift)
        preds_[target == self.ignore_idx] = 0.
        preds_ = preds_.view(B, -1)  # [B, L*L]
        val, idx = torch.topk(preds_, L, dim=-1)  # [B, L]
        
        target_ = target.view(B, -1)  # [B, L*L]
        target_ = torch.gather(target_, dim=1, index=idx)  # [B, L]
        
        preds_ = torch.argmax(preds, dim=1)  # [B, L, L]
        preds_ = preds_.view(B, -1)  # [B, L*L]
        preds_ = torch.gather(preds_, dim=1, index=idx)  # [B, L]
        
        tp = torch.logical_and(preds_ == 1, target_ == 1).sum()
        fp = torch.logical_and(preds_ == 1, target_ == 0).sum()

        self.tp += tp
        self.fp += fp

    def compute(self) -> torch.Tensor:
        """
        Computes binary top-L precision.

        Returns:
            torch.Tensor: Top-L precision.
        """
        
        return self.tp.float() / (self.tp.float() + self.fp.float())


class BinaryConfusionMatrix(Metric):
    def __init__(self, dist_sync_on_step: bool = False) -> None:
        """
        Initializes confusion matrix metric for binary classification with ignore index support.

        Args:
            dist_sync_on_step (bool, optional): Synchronize metric state across processes at each forward() before returning the value at the step. Defaults to False.
        """
        
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates internal metric states by comparing predicted and target values.
        Ignored target values should be set negative.

        Args:
            preds (torch.Tensor): Predictions [B, 2, L, L].
            target (torch.Tensor): Targets [B, L, L].
        """
        
        assert preds.shape[1] == 2
        preds_ = preds.permute(0, 2, 3, 1)  # [B, L, L, 2]
        preds_ = torch.argmax(preds_, dim=-1)  # [B, L, L]
        assert preds_.size() == target.size()
        
        tp = torch.logical_and(preds_ == 1, target == 1).sum()
        fp = torch.logical_and(preds_ == 1, target == 0).sum()
        tn = torch.logical_and(preds_ == 0, target == 0).sum()
        fn = torch.logical_and(preds_ == 0, target == 1).sum()

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self) -> torch.Tensor:
        """
        Computes confusion matrix for binary classification.

        Returns:
            torch.Tensor: [[TP, FN], [FP, TN]]
        """
        
        return torch.tensor([[self.tp, self.fn], [self.fp, self.tn]])