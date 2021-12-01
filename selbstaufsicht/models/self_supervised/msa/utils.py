from functools import partial
import torch
import collections
from torch.nn import CrossEntropyLoss
from torch.nn import ModuleDict
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding
from selbstaufsicht.models.self_supervised.msa.transforms import RandomMSACropping, RandomMSASubsampling, RandomMSAShuffling
from selbstaufsicht.modules import NT_Xent_Loss, Accuracy
from .modules import InpaintingHead, JigsawHead, ContrastiveHead

# NOTE mask and padding tokens can not be reconstructed


def get_tasks(tasks,
              dim,
              subsample_depth=5,
              subsample_mode='uniform',
              crop=50,
              masking='token',
              p_mask=0.15,
              jigsaw_partitions=3,
              jigsaw_classes=4,
              jigsaw_padding_token=-1,
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
        RandomMSASubsampling(subsample_depth, contrastive=contrastive, mode=subsample_mode),
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
    if 'jigsaw' in tasks:
        head = JigsawHead(dim, jigsaw_classes)
        task_heads['jigsaw'] = head
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


class MSACollator():
    def __init__(self, msa_padding_token, inpainting_mask_padding_token=0, jigsaw_padding_token=-1):
        self.collate_fn = {
            'msa': partial(_pad_collate_nd, pad_val=msa_padding_token, need_padding_mask=True),
            'mask': partial(_pad_collate_nd, pad_val=inpainting_mask_padding_token),
            'aux_features': _pad_collate_nd,
            'aux_features_contrastive': _pad_collate_nd,
            'inpainting': _flatten_collate,
            'jigsaw': partial(_pad_collate_nd, pad_val=jigsaw_padding_token),
            'contrastive': partial(_pad_collate_nd, pad_val=msa_padding_token, need_padding_mask=True),
        }

    def __call__(self, batch):
        """
        batch: list of tuples of dicts: e.g. [({input1}, {target1}), ({input2}, {target2})]
        input contains: {'msa': tensor, optional 'mask': tensor, 'aux_features': tensor, 'contrastive': tensor}
        target contains one or more of {'inpainting': 1dtensor, 'jigsaw': 1dtensor}
        """

        first_sample = batch[0]
        if all([isinstance(item, collections.abc.Mapping) for item in first_sample]):
            result = tuple(self._collate_dict(idx, item, batch) for idx, item in enumerate(first_sample))
            return result
        else:
            raise
            # return torch.utils.data._utils.collate.default_collate(batch)
    
    def _collate_dict(self, item_idx, item, batch):
        
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


def _pad_collate_nd(batch, pad_val=0, need_padding_mask=False):
    '''
    batch: sequence of nd tensors that may have different dimensions
    '''
    
    kronecker_delta = lambda i, j: 1 if i == j else 0

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


def _flatten_collate(batch):
    '''
    batch: sequence of 1d tensors, that may be of different length
    '''
    
    # optimization taken from default collate_fn, using shared memory to avoid a copy when passing data to the main process
    out = None
    if torch.utils.data.get_worker_info() is not None:
        elem = batch[0]
        numel = sum(x.numel() for x in batch)
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    return torch.cat(batch, 0, out=out)