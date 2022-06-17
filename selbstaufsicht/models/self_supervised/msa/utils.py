from functools import partial
import torch
from typing import Dict, List, Tuple, Union

from torch.nn import CrossEntropyLoss, MSELoss
from torchmetrics import MeanAbsoluteError
from torch.nn import Module, ModuleDict
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index, nonstatic_mask_tokens
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding
from selbstaufsicht.models.self_supervised.msa.transforms import MSACropping, MSASubsampling, RandomMSAShuffling, MSAboot
from selbstaufsicht.models.self_supervised.msa.transforms import DistanceFromChain, ContactFromDistance
from selbstaufsicht.modules import NT_Xent_Loss, Accuracy, EmbeddedJigsawAccuracy, EmbeddedJigsawLoss, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from .modules import InpaintingHead, JigsawHead, ContrastiveHead

# NOTE mask and padding tokens can not be reconstructed


def get_tasks(tasks: List[str],
              dim: int,
              subsample_depth: int = 5,
              subsample_mode: str = 'uniform',
              crop_size: int = 50,
              crop_mode: str = 'random-dependent',
              masking: str = 'token',
              p_mask: float = 0.15,
              p_mask_static: float = 0.8,
              p_mask_nonstatic: float = 0.1,
              p_mask_unchanged: float = 0.1,
              jigsaw_partitions: int = 3,
              jigsaw_classes: int = 4,
              jigsaw_padding_token: int = -1,
              jigsaw_linear: bool = True,
              jigsaw_euclid_emb: torch.Tensor = None,
              jigsaw_delimiter: bool = True,
              simclr_temperature: float = 100.,
              jigsaw_boot_ratio: float = 0.5,
              per_token: bool = False,
              boot_same: bool = False,
              frozen: bool = False,
              seq_dist: bool = False
              ) -> Tuple[transforms.SelfSupervisedCompose, Dict[str, Module], Dict[str, Module], Dict[str, ModuleDict]]:
    """
    Configures task heads, losses, data transformations, and evaluation metrics for given task parameters.

    Args:
        tasks (List[str]): Upstream tasks to be performed.
        dim (int): Embedding dimensionality.
        subsample_depth (int, optional): Number of subsampled sequences per MSA. Defaults to 5.
        subsample_mode (str, optional): Subsampling mode. Defaults to 'uniform'.
        crop_size (int, optional): Maximum uncropped sequence length. Defaults to 50.
        crop_mode (str, optional): Cropping mode. Defaults to 'random-dependent'.
        masking (str, optional): Masking mode for inpainting. Defaults to 'token'.
        p_mask (float, optional): Masking probability for inpainting. Defaults to 0.15.
        p_mask_static (float, optional): Conditional probability for static inpainting, if masked. Defaults to 0.8.
        p_mask_nonstatic (float, optional): Conditional probability for nonstatic inpainting, if masked. Defaults to 0.1.
        p_mask_unchanged (float, optional): Conditional probability for no change, if masked. Defaults to 0.1.
        jigsaw_partitions (int, optional): Number of shuffled partitions for jigsaw. Defaults to 3.
        jigsaw_classes (int, optional): Number of allowed permutations for jigsaw. Defaults to 4.
        jigsaw_padding_token (int, optional): Special token that indicates padded sequences in the jigsaw label. Defaults to -1.
        jigsaw_linear (bool, optional): if True linear head, otherwise two layer MLP. Defaults to True.
        jigsaw_euclid_emb (torch.Tensor, optional): Euclidean embedding of the discrete permutation metric. Defaults to None.
        jigsaw_delimiter (bool, optional): Whether delimiter token is inserted between jigsaw partitions. Defaults to True.
        simclr_temperature (float, optional): Distillation temperatur for the SimCLR loss of contrastive learning. Defaults to 100..

    Raises:
        ValueError: Unknown upstream task.

    Returns:
        Tuple[transforms.SelfSupervisedCompose, Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, ModuleDict], Dict[str, ModuleDict]]:
        Composition of preprocessing transforms; head modules for upstream tasks;
        loss functions for upstream tasks; further training/validation metrics for upstream tasks
    """

    if not set(tasks) <= {'inpainting', 'jigsaw', 'contrastive', 'jigsaw_boot'}:
        raise ValueError('unknown task id')

    contrastive = False
    if 'contrastive' in tasks:
        contrastive = True

    transformslist = [
        MSASubsampling(subsample_depth, contrastive=contrastive, mode=subsample_mode),
        MSACropping(crop_size, contrastive=contrastive, mode=crop_mode),
        MSATokenize(rna2index)]

    if 'jigsaw' in tasks:
        delimiter_token = rna2index['DELIMITER_TOKEN'] if jigsaw_delimiter else None
        transformslist.append(
            RandomMSAShuffling(
                delimiter_token=delimiter_token,
                num_partitions=jigsaw_partitions,
                num_classes=jigsaw_classes,
                euclid_emb=jigsaw_euclid_emb,
                contrastive=contrastive,frozen=frozen)
        )
    if 'inpainting' in tasks:
        transformslist.append(
            RandomMSAMasking(
                p=p_mask,
                p_static=p_mask_static,
                p_nonstatic=p_mask_nonstatic,
                p_unchanged=p_mask_unchanged,
                mode=masking,
                static_mask_token=rna2index['MASK_TOKEN'],
                nonstatic_mask_tokens=nonstatic_mask_tokens,
                contrastive=contrastive)
        )

    if 'jigsaw_boot' in tasks:
        transformslist.append(MSAboot(ratio=jigsaw_boot_ratio, per_token=per_token, boot_same=boot_same, seq_dist=seq_dist))

    transformslist.append(ExplicitPositionalEncoding())

    transform = transforms.SelfSupervisedCompose(transformslist)

    train_metrics = ModuleDict()
    val_metrics = ModuleDict()

    task_heads = ModuleDict()
    task_losses = dict()
    if 'jigsaw' in tasks:
        if jigsaw_euclid_emb is not None:
            jigsaw_classes = jigsaw_euclid_emb.shape[1]
            train_acc_metric = EmbeddedJigsawAccuracy(jigsaw_euclid_emb, ignore_value=jigsaw_padding_token)
            val_acc_metric = EmbeddedJigsawAccuracy(jigsaw_euclid_emb, ignore_value=jigsaw_padding_token)
            loss_fn = EmbeddedJigsawLoss(ignore_value=jigsaw_padding_token)
        else:
            if frozen:
                train_acc_metric = Accuracy(class_dim=-1, ignore_index=jigsaw_padding_token)
                val_acc_metric = Accuracy(class_dim=-1, ignore_index=jigsaw_padding_token)
            else:
                train_acc_metric = Accuracy(class_dim=-2, ignore_index=jigsaw_padding_token)
                val_acc_metric = Accuracy(class_dim=-2, ignore_index=jigsaw_padding_token)
            loss_fn = CrossEntropyLoss(ignore_index=jigsaw_padding_token)

        head = JigsawHead(dim, jigsaw_classes, proj_linear=jigsaw_linear, euclid_emb=jigsaw_euclid_emb is not None, frozen=frozen)
        task_heads['jigsaw'] = head
        task_losses['jigsaw'] = loss_fn
        train_metrics['jigsaw'] = ModuleDict({'acc': train_acc_metric})
        val_metrics['jigsaw'] = ModuleDict({'acc': val_acc_metric})
    if 'inpainting' in tasks:
        # NOTE never predict mask token or padding token
        head = InpaintingHead(dim, len(rna2index) - 2)
        task_heads['inpainting'] = head

        task_losses['inpainting'] = CrossEntropyLoss()
        train_metrics['inpainting'] = ModuleDict({'acc': Accuracy()})
        val_metrics['inpainting'] = ModuleDict({'acc': Accuracy()})
    if 'contrastive' in tasks:
        head = ContrastiveHead(dim)
        task_heads['contrastive'] = head
        task_losses['contrastive'] = NT_Xent_Loss(simclr_temperature)
        train_metrics['contrastive'] = ModuleDict({})
        val_metrics['contrastive'] = ModuleDict({})
    if 'jigsaw_boot' in tasks:
        if not per_token:
            head = JigsawHead(dim, num_classes=2, proj_linear=True, euclid_emb=False, boot=True)
            task_heads['jigsaw_boot'] = head
            task_losses['jigsaw_boot'] = CrossEntropyLoss()
            train_metrics['jigsaw_boot'] = ModuleDict({'acc': Accuracy()})
            val_metrics['jigsaw_boot'] = ModuleDict({'acc': Accuracy()})
        else:
            if seq_dist:
                head = JigsawHead(dim, num_classes=1, proj_linear=True, euclid_emb=False, boot=True, seq_dist=True)
                task_heads['jigsaw_boot'] = head
                task_losses['jigsaw_boot'] = MSELoss()
                train_metrics['jigsaw_boot'] = ModuleDict({'mae': MeanAbsoluteError()})
                val_metrics['jigsaw_boot'] = ModuleDict({'mae': MeanAbsoluteError()})
            else:
                head = InpaintingHead(dim, 2, boot=True)
                task_heads['jigsaw_boot'] = head
                task_losses['jigsaw_boot'] = CrossEntropyLoss()
                train_metrics['jigsaw_boot'] = ModuleDict({'acc': Accuracy()})
                val_metrics['jigsaw_boot'] = ModuleDict({'acc': Accuracy()})

    return transform, task_heads, task_losses, train_metrics, val_metrics


def get_downstream_transforms(subsample_depth, subsample_mode: str = 'uniform', jigsaw_partitions: int = 0, threshold: float = 4., inference=False, device=None):
    transformslist = [
        MSASubsampling(subsample_depth, mode=subsample_mode),
        MSATokenize(rna2index)]
    if jigsaw_partitions > 0:
        transformslist.append(
            RandomMSAShuffling(
                delimiter_token=rna2index['DELIMITER_TOKEN'],
                num_partitions=jigsaw_partitions,
                num_classes=1)
        )
    transformslist.append(ExplicitPositionalEncoding())
    if not inference:
        transformslist.append(DistanceFromChain(device=device))
        transformslist.append(ContactFromDistance(threshold))
    downstream_transform = transforms.SelfSupervisedCompose(transformslist)

    return downstream_transform


def get_downstream_metrics():
    train_metrics = ModuleDict()
    val_metrics = ModuleDict()
    test_metrics = ModuleDict()

    train_metrics['contact'] = ModuleDict({'acc': Accuracy(class_dim=1, ignore_index=-1), 'topLprec': BinaryPrecision(),
                                           'topLprec_coconet': BinaryPrecision(treat_all_preds_positive=True),
                                           'topLprec_unreduced': BinaryPrecision(reduce=False),
                                           'Global_precision': BinaryPrecision(k=-1), 'Global_recall': BinaryRecall(),
                                           'Global_F1score': BinaryF1Score(), 'confmat': BinaryConfusionMatrix(),
                                           'confmat_unreduced': BinaryConfusionMatrix(reduce=False)})

    val_metrics['contact'] = ModuleDict({'acc': Accuracy(class_dim=1, ignore_index=-1), 'topLprec': BinaryPrecision(),
                                         'topLprec_coconet': BinaryPrecision(treat_all_preds_positive=True),
                                         'topLprec_unreduced': BinaryPrecision(reduce=False),
                                         'Global_precision': BinaryPrecision(k=-1), 'Global_recall': BinaryRecall(),
                                         'Global_F1score': BinaryF1Score(), 'confmat': BinaryConfusionMatrix(),
                                         'confmat_unreduced': BinaryConfusionMatrix(reduce=False)})

    test_metrics['contact'] = ModuleDict(
            {
                'acc': Accuracy(class_dim=1, ignore_index=-1),
                'topLprec': BinaryPrecision(),
                'topLprec_coconet': BinaryPrecision(treat_all_preds_positive=True),
                'topLprec_unreduced': BinaryPrecision(reduce=False),
                'Global_precision': BinaryPrecision(k=-1), 'Global_recall': BinaryRecall(),
                'Global_F1score': BinaryF1Score(), 'confmat': BinaryConfusionMatrix(),
                'confmat_unreduced': BinaryConfusionMatrix(reduce=False)
            })
    return train_metrics, val_metrics, test_metrics


class MSACollator():
    def __init__(self, msa_padding_token: int, inpainting_mask_padding_token: int = 0, jigsaw_padding_token: int = -1,frozen: bool = False) -> None:
        """
        Initializes MSA collator.

        Args:
            msa_padding_token (int): Special token that is used for padding of MSA with different shapes.
            inpainting_mask_padding_token (int, optional): Special token that is used for padding of boolean MSA inpainting masking masks with different shapes. Defaults to 0.
            jigsaw_padding_token (int, optional): Special token that indicates padded sequences in the jigsaw label. Defaults to -1.
        """

        self.collate_fn = {
            'msa': partial(_pad_collate_nd, pad_val=msa_padding_token, need_padding_mask=True),
            'mask': partial(_pad_collate_nd, pad_val=inpainting_mask_padding_token),
            'aux_features': _pad_collate_nd,
            'aux_features_contrastive': _pad_collate_nd,
            'inpainting': _flatten_collate,
            'jigsaw': partial(_pad_collate_nd, pad_val=jigsaw_padding_token),
            'contrastive': partial(_pad_collate_nd, pad_val=msa_padding_token, need_padding_mask=True),
            'jigsaw_boot': _flatten_collate
            }

        if frozen:
            self.collate_fn['jigsaw'] = _flatten_collate

    def __call__(self, batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs collation of batch data, i.e., shapes of different batch items are aligned by padding and they are concatenated in a new batch dimension for each tensor.

        Args:
            batch (List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
            Batch data: Each batch item consists of input and target data, which in turn contain several tensors.
            Input data contains {'msa': tensor, optional 'mask': tensor, 'aux_features': tensor, 'contrastive': tensor}.
            Target data contains one or more of {'inpainting': 1dtensor, 'jigsaw': 1dtensor}

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Collated batch data: Input and target data contain several tensors, which include a batch dimension.
        """

        return tuple(self._collate_dict(idx, item, batch) for idx, item in enumerate(batch[0]))

    def _collate_dict(self, item_idx: int, item: Dict[str, torch.Tensor], batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Performs collation of batch data for a single data item, i.e., either input or target data.

        Args:
            item_idx (int): Index of the data item.
            item (Dict[str, torch.Tensor]): Data item, containing several tensors.
            batch (List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
            Full batch data: Each batch item consists of input and target data, which in turn contain several tensors.

        Returns:
            Dict[str, torch.Tensor]: Collated batch data for the given data item, i.e., it contains several tensors, which include a batch dimension.
        """

        out = {}

        for key in item:
            collate_result = self.collate_fn[key]([sample[item_idx][key] for sample in batch])

            if key in ['msa', 'contrastive']:
                collate_result, padding_mask = collate_result
                if key == 'msa':
                    out['padding_mask'] = padding_mask
                elif key == 'contrastive':
                    out['padding_mask_contrastive'] = padding_mask
            out[key] = collate_result

        return out


def _pad_collate_nd(batch: List[torch.Tensor], pad_val: int = 0, need_padding_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pads and concatenates a list of possibly differently shaped n-dimensional tensors along a new batch dimension to a single tensor.

    Args:
        batch (List[torch.Tensor]): List of possibly differently shaped n-dimensional tensors.
        pad_val (int, optional): Padding value. Defaults to 0.
        need_padding_mask (bool, optional): Whether the padding mask is needed as an output. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Padded and concatenated tensors; boolean padding mask (optional)
    """

    def kronecker_delta(i: int, j: int) -> int:
        """
        Kronecker delta, which is 1 for two equal numbers and 0 for two different numbers.

        Args:
            i (int): First number.
            j (int): Second number.

        Returns:
            int: Either 1 oder 0.
        """

        if i == j:
            return 1
        return 0

    B = len(batch)
    n_dims = batch[0].dim()
    dims = [max([sample.size(dim) for sample in batch]) for dim in range(n_dims)]

    # optimization taken from default collate_fn, using shared memory to avoid a copy when passing data to the main process
    if torch.utils.data.get_worker_info() is not None:
        elem = batch[0]
        numel = B * torch.tensor(dims).prod().item()
        storage = elem.storage()._new_shared(numel)
        out_shm = elem.new(storage)
        out_shm[:] = pad_val
        out = out_shm.view(B, *dims)
        if need_padding_mask:
            dummy_bool = torch.tensor(False)
            storage_mask = dummy_bool.storage()._new_shared(numel)
            out_mask_shm = dummy_bool.new(storage_mask)
            out_mask = out_mask_shm.view(B, *dims)
    else:
        out = torch.full((B, *dims), pad_val, dtype=batch[0].dtype)
        if need_padding_mask:
            out_mask = torch.zeros((B, *dims), dtype=torch.bool)

    for idx, sample in enumerate(batch):
        inplace_slice = (idx, ) + tuple(slice(sample_dim) for sample_dim in sample.size())
        insert_slice = (slice(None),) * n_dims
        out[inplace_slice] = sample[insert_slice]
        if need_padding_mask:
            mask_slices = [(idx, ) + tuple(slice(kronecker_delta(dim_i, dim_j) * sample.size(dim_i), None) for dim_i in range(n_dims)) for dim_j in range(n_dims)]
            for mask_slice in mask_slices:
                out_mask[mask_slice] = True

    if need_padding_mask:
        return out, out_mask
    else:
        return out


def _flatten_collate(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenates a list of possibly differently shaped 1-dimensional tensors along the existing dimension to a single 1-dimensional tensor.

    Args:
        batch (List[torch.Tensor]): List of possibly differently shaped 1-dimensional tensors.

    Returns:
        torch.Tensor: Concatenated 1-dimensional tensor.
    """

    # optimization taken from default collate_fn, using shared memory to avoid a copy when passing data to the main process
    out = None
    if torch.utils.data.get_worker_info() is not None:
        elem = batch[0]
        numel = sum(x.numel() for x in batch)
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    return torch.cat(batch, 0, out=out)
