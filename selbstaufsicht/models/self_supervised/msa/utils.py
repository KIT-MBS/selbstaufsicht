import torch
import collections
from torch.nn import CrossEntropyLoss
from torch.nn import ModuleDict
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding
from selbstaufsicht.models.self_supervised.msa.transforms import RandomMSACropping, RandomMSASubsampling, RandomMSAShuffling
from selbstaufsicht.modules import NT_Xent_Loss
from .modules import InpaintingHead, JigsawHead, ContrastiveHead
from torchmetrics import Accuracy

# NOTE mask and delimiter tokens can not be reconstructed


def get_tasks(tasks,
              dim,
              subsample_depth=5,
              crop=50,
              masking='token',
              p_mask=0.15,
              jigsaw_partitions=3,
              jigsaw_classes=4,
              simclr_temperature=100.,
              ):
    """
    configures task heads, losses, data transformations, and evaluation metrics given task parameters
    """
    if not set(tasks) <= {'inpainting', 'jigsaw', 'contrastive'}:
        raise ValueError('unknown task id')
    if masking != 'token':
        raise NotImplementedError('only token masking')

    contrastive = False
    if 'contrastive' in tasks:
        contrastive = True

    transformslist = [
        RandomMSASubsampling(subsample_depth, contrastive=contrastive),
        RandomMSACropping(crop, contrastive=contrastive),
        MSATokenize(rna2index)]

    if 'jigsaw' in tasks:
        transformslist.append(
            RandomMSAShuffling(
                delimiter_token=rna2index['DELIMITER_TOKEN'],
                num_partitions=jigsaw_partitions,
                num_classes=jigsaw_classes,
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

    transform = transforms.Compose(transformslist)

    metrics = ModuleDict()

    task_heads = ModuleDict()
    task_losses = dict()
    # TODO always include 'no transformation'
    if 'jigsaw' in tasks:
        head = JigsawHead(dim, jigsaw_classes)
        task_heads['jigsaw'] = head
        task_losses['jigsaw'] = CrossEntropyLoss()
        metrics['jigsaw'] = ModuleDict({'acc': Accuracy()})
    if 'inpainting' in tasks:
        head = InpaintingHead(dim, len(rna2index) - 1)  # NOTE never predict mask token
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


class MSACollator():
    def __init__(self):
        self.collate_dict = {
            'msa': _pad_collate_nd,
            'mask': _pad_collate_nd,
            'aux_features': _pad_collate_nd,
            'inpainting': _flatten_collate,
            'jigsaw': torch.utils.data._utils.collate.default_collate,
            'contrastive': _pad_collate_nd,
        }

    def __call__(self, batch):
        """
        x: list of tuples of dicts: e.g. [({input1}, {target1}), ({input2}, {target2})]
        input contains: {'msa': tensor, optional 'mask': tensor, 'aux_features': tensor}
        target contains one or more of {'inpainting': 1dtensor, 'jigsaw': int, 'contrastive': tensor}
        """

        first_sample = batch[0]
        if all([isinstance(item, collections.abc.Mapping) for item in first_sample]):
            return tuple({key: self.collate_dict[key]([sample[idx][key] for sample in batch]) for key in item} for idx, item in enumerate(first_sample))
        else:
            return torch.utils.data._utils.collate.default_collate(batch)


def _pad_collate_nd(batch):
    '''
    batch: sequence of nd tensors, that may have different dimensions
    '''

    B = len(batch)
    n_dims = batch[0].dim()
    dims = [max([sample.size(dim) for sample in batch]) for dim in range(n_dims)]

    out = torch.zeros((B, *dims), dtype=batch[0].dtype)
    for idx, sample in enumerate(batch):
        inplace_slices = (idx, ) + tuple(slice(sample_dim) for sample_dim in sample.size())
        insert_slices = (slice(None),) * n_dims
        out[inplace_slices] = sample[insert_slices]

    return out


def _flatten_collate(batch):
    '''
    batch: sequence of 1d tensors, that may be of different length
    '''
    # TODO optimize for multiple workers
    return torch.cat(batch, 0)
