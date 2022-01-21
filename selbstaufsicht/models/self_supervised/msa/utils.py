from functools import partial
import torch
from typing import Dict, List, Tuple, Union

from torch.nn import CrossEntropyLoss
from torch.nn import Module, ModuleDict
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding
from selbstaufsicht.models.self_supervised.msa.transforms import MSACropping, MSASubsampling, RandomMSAShuffling
from selbstaufsicht.models.self_supervised.msa.transforms import DistanceFromChain, ContactFromDistance
from selbstaufsicht.modules import NT_Xent_Loss, Accuracy, EmbeddedJigsawLoss
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
              jigsaw_partitions: int = 3,
              jigsaw_classes: int = 4,
              jigsaw_padding_token: int = -1,
              jigsaw_linear: bool = True,
              jigsaw_euclid_emb: torch.Tensor = None,
              simclr_temperature: float = 100.,
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
        jigsaw_partitions (int, optional): Number of shuffled partitions for jigsaw. Defaults to 3.
        jigsaw_classes (int, optional): Number of allowed permutations for jigsaw. Defaults to 4.
        jigsaw_padding_token (int, optional): Special token that indicates padded sequences in the jigsaw label. Defaults to -1.
        jigsaw_linear (bool, optional): if True linear head, otherwise two layer MLP. Defaults to True.
        jigsaw_euclid_emb (bool, optional): Euclidean embedding of the discrete permutation metric. Defaults to None.
        simclr_temperature (float, optional): Distillation temperatur for the SimCLR loss of contrastive learning. Defaults to 100..

    Raises:
        ValueError: Unknown upstream task.

    Returns:
        Tuple[transforms.SelfSupervisedCompose, Dict[str, nn.Module], Dict[str, nn.Module], Dict[str, ModuleDict]]:
        Composition of preprocessing transforms; head modules for upstream tasks;
        loss functions for upstream tasks; further metrics for upstream tasks
    """

    if not set(tasks) <= {'inpainting', 'jigsaw', 'contrastive'}:
        raise ValueError('unknown task id')

    contrastive = False
    if 'contrastive' in tasks:
        contrastive = True

    transformslist = [
        MSASubsampling(subsample_depth, contrastive=contrastive, mode=subsample_mode),
        MSACropping(crop_size, contrastive=contrastive, mode=crop_mode),
        MSATokenize(rna2index)]

    if 'jigsaw' in tasks:
        transformslist.append(
            RandomMSAShuffling(
                delimiter_token=rna2index['DELIMITER_TOKEN'],
                num_partitions=jigsaw_partitions,
                num_classes=jigsaw_classes,
                euclid_emb=jigsaw_euclid_emb,
                contrastive=contrastive)
        )
    if 'inpainting' in tasks:
        transformslist.append(
            RandomMSAMasking(
                p=p_mask,
                mode=masking,
                mask_token=rna2index['MASK_TOKEN'],
                contrastive=contrastive)
        )

    transformslist.append(ExplicitPositionalEncoding())

    transform = transforms.SelfSupervisedCompose(transformslist)

    metrics = ModuleDict()

    task_heads = ModuleDict()
    task_losses = dict()
    if 'jigsaw' in tasks:
        head = JigsawHead(dim, jigsaw_classes, proj_linear=jigsaw_linear)
        task_heads['jigsaw'] = head
        if jigsaw_euclid_emb is not None:
            task_losses['jigsaw'] = EmbeddedJigsawLoss(ignore_value=jigsaw_padding_token)
        else:
            task_losses['jigsaw'] = CrossEntropyLoss(ignore_index=jigsaw_padding_token)
        metrics['jigsaw'] = ModuleDict({'acc': Accuracy(class_dim=-2, ignore_index=jigsaw_padding_token)})
    if 'inpainting' in tasks:
        head = InpaintingHead(dim, len(rna2index) - 2)  # NOTE never predict mask token or padding token
        task_heads['inpainting'] = head

        task_losses['inpainting'] = CrossEntropyLoss()
        metrics['inpainting'] = ModuleDict({'acc': Accuracy()})
    if 'contrastive' in tasks:
        head = ContrastiveHead(dim)
        task_heads['contrastive'] = head
        task_losses['contrastive'] = NT_Xent_Loss(simclr_temperature)
        # TODO
        metrics['contrastive'] = ModuleDict({})

    return transform, task_heads, task_losses, metrics


def get_downstream_transforms(subsample_depth, jigsaw_partitions: int = 0, threshold: float = 4.):
    # TODO better subsampling
    transformslist = [
        MSASubsampling(subsample_depth, mode='uniform'),
        MSATokenize(rna2index)]
    if jigsaw_partitions > 0:
        transformslist.append(
            RandomMSAShuffling(
                delimiter_token=rna2index['DELIMITER_TOKEN'],
                num_partitions=jigsaw_partitions,
                num_classes=1)
        )
    transformslist.append(ExplicitPositionalEncoding())
    transformslist.append(DistanceFromChain())
    transformslist.append(ContactFromDistance(threshold))
    downstream_transform = transforms.SelfSupervisedCompose(transformslist)

    return downstream_transform


class MSACollator():
    def __init__(self, msa_padding_token: int, inpainting_mask_padding_token: int = 0, jigsaw_padding_token: int = -1) -> None:
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
        }

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
