from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSA2Tensor, RandomMSAColumnMasking
from .modules import InpaintingHead


def get_tasks(task, dim, **kwargs):
    if task != 'inpainting':
        raise NotImplementedError
    if kwargs['subsampling'] != 'random':
        raise NotImplementedError

    # TODO different masking strategies
    transform = transforms.Compose(
            [
                MSA2Tensor(rna2index),
                OneHot,
                RandomMSAMasking(p=0.15),
                ExplicitPositional(),
            ])
    head = InpaintingHead(dim, kwargs['num_classes'])

    return transform, head
