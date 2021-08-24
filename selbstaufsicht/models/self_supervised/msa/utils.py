from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize,  RandomMSAMasking, ExplicitPositionalEncoding
from .modules import InpaintingHead


def get_tasks(task, dim, **kwargs):
    # TODO other tasks
    if task != 'inpainting':
        raise NotImplementedError
    if kwargs['subsampling'] != 'random':
        raise NotImplementedError

    masking = kwargs.get(['masking'], 'uniform')
    p_mask = kwargs.get(['p_mask'], 0.15)
    transform = transforms.Compose(
            [
                MSATokenize(rna2index),
                RandomMSAMasking(p=p_mask, mode=masking, mask_token=rna2index['MASK_TKEN']),
                ExplicitPositionalEncoding(),
            ])
    head = InpaintingHead(dim, kwargs['num_classes'])

    return transform, head
