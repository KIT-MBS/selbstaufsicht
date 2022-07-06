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


def sync_(tensor):
    """
    gathers the tensors in the global batch on the same node into one list
    """
    gathered = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, tensor)
    gathered = [t.clone().to(device=tensor.device) for t in gathered]

    return gathered


# NOTE this would be more flexible, but since all_gather needs tensors of identical shape, we need to use all_gather_objects (or point to point communication) which does not send the tensor data
# class SyncFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, tensor):
#         ctx.batch_size = tensor.size(0)
#
#         gathered = [None for _ in range(torch.distributed.get_world_size())]
#         torch.distributed.all_gather_object(gathered, tensor)
#         # NOTE this does not work for variably sizes tensors
#         # gathered_sizes = (t.size() for t in gathered)
#         # gathered = [torch.zeros(s, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device) for s in gathered_sizes]
#         # torch.distributed.all_gather(gathered, tensor)
#         gathered = [t.c]
#
#         return gathered
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         print('rank', torch.distributed.get_rank(), grad_output)
#         raise
#         # grad_input = [x.clone() for x in grad_output]
#
#         return


class SequenceNTXentLoss(nn.Module):
    def __init__(self, temperature: float, eps: float = 1e-6) -> None:
        super(SequenceNTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, x: torch.Tensor, y) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): output of a contrastive head [B, E, D]
            y: dummy variable

        Returns:
            torch.Tensor: loss
        """
        if y is not None:
            raise ValueError('unexpected item in the bugging area')

        # NOTE currently works only for batch_size==1
        assert x.size(0) == 1
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed:
            # x_dist = SyncFunction.apply(x)
            x_dist = sync_(x)

        else:
            raise RuntimeError('Contrastive needs multiple samples, since batch size == 1 distributed.')

        x_neg = torch.cat([t for i, t in enumerate(x_dist) if i != torch.distributed.get_rank()], dim=1)  # [sum(E_j|j!=rank), D]
        x_neg = x_neg.squeeze()
        x = x.squeeze()  # [E_rank, D]

        cov = torch.mm(x, x_neg.t().contiguous())  # [E_rank, sumj!=i(E_j)]
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)  # [E_rank]
        neg = torch.clamp(neg, min=self.eps)
        neg = torch.unsqueeze(neg, 1)

        pos = torch.exp(torch.mm(x, x.t().contiguous()))  # [E_rank, E_rank]

        loss = -torch.log(pos / (neg + self.eps)).mean()
        return loss


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


class SigmoidCrossEntropyLoss(nn.NLLLoss):
    def __init__(self, weight: torch.Tensor = None, ignore_index: int = -1, reduction='mean'):
        """
        Initializes Sigmoid Cross Entropy Loss with ignore-index support.
        This replaces the softmax operation of usual cross entropy loss by sigmoid.

        Args:
            weight (torch.Tensor, optional): Weights, by which each class is re-scaled in the loss computation. Defaults to None.
            ignore_index (int, optional): Target value which is ignored in comparison. Defaults to -1.
            reduction (str, optional): Reduction mode: mean, sum. Defaults to \"mean\".
        """

        super(SigmoidCrossEntropyLoss, self).__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.log_sigmoid = torch.nn.LogSigmoid()

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes SigmoidCrossEntropyLoss for given predictions and target data.

        Args:
            preds (torch.Tensor): Predictions (one-hot-encoded with class-dim 1).
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: SigmoidCrossEntropyLoss.
        """

        preds_ = self.log_sigmoid(preds)
        return super().forward(preds_, target)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2., weight: torch.Tensor = None, ignore_index: int = -1, reduction='mean'):
        """
        Initializes Binary Focal Negative Log Likelihood Loss with ignore-index support.

        Args:
            gamma (float, optional): Exponent of the modulating factor. Defaults to 2..
            weight (torch.Tensor, optional): Weights, by which each class is re-scaled in the loss computation. Defaults to None.
            ignore_index (int, optional): Target value which is ignored in comparison. Defaults to -1.
            reduction (str, optional): Reduction mode: mean, sum. Defaults to \"mean\".
        """

        super(BinaryFocalLoss, self).__init__()
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
        Computes BinaryFocalLoss for given predictions and target data (cf. https://arxiv.org/abs/1708.02002v2).

        Args:
            preds (torch.Tensor): Predictions (one-hot-encoded with class-dim 1).
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: BinaryFocalLoss.
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
        preds_ = torch.sigmoid(preds)
        preds_ = preds_.permute(0, *(idx for idx in range(2, preds_.ndim)), 1)
        preds_ = preds_[mask, :]
        preds_log = nn.functional.logsigmoid(preds)
        preds_log = preds_log.permute(0, *(idx for idx in range(2, preds_log.ndim)), 1)
        preds_log = preds_log[mask, :]

        # compute focal loss
        loss = torch.zeros(target.shape, device=target.device)
        loss[mask] = -(self.weight[0] * preds_[:, 1] ** self.gamma * (1 - target_) * preds_log[:, 0] +
                       self.weight[1] * preds_[:, 0] ** self.gamma * target_ * preds_log[:, 1])

        # perform reduction, if needed
        if self.reduction in ('sum', 'mean'):
            loss = loss.sum()
        if self.reduction == 'mean':
            # divide by sum of applied class weights in order to normalize w.r.t. class weights
            loss /= self.weight[target_].sum()

        return loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8, ignore_index: int = -1, reduction='mean'):
        """
        Initializes Dice Negative Log Likelihood Loss with ignore-index support.

        Args:
            epsilon (float, optional): Additive for numerical stabilization. Defaults to 1e-8.
            ignore_index (int, optional): Target value which is ignored in comparison. Defaults to -1.
            reduction (str, optional): Reduction mode: mean, sum. Defaults to \"mean\".
        """

        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes DiceLoss for given predictions and target data (cf. https://arxiv.org/abs/1707.03237).

        Args:
            preds (torch.Tensor): Predictions (one-hot-encoded with class-dim 1).
            target (torch.Tensor): Target data.

        Returns:
            torch.Tensor: DiceLoss.
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
        preds_ = torch.sigmoid(preds)
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
