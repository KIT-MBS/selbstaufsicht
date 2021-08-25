from torch.nn import CrossEntropyLoss
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize,  RandomMSAMasking, ExplicitPositionalEncoding
from .modules import InpaintingHead

# NOTE mask and delimiter tokens can not be reconstructed
num_special_tokens = 2


def get_tasks(task, dim, **kwargs):
    # TODO other tasks
    if task != ['inpainting']:
        raise NotImplementedError('only inpainting')
    if kwargs['subsampling'] != 'token':
        raise NotImplementedError('only token masking')

    masking = kwargs.get('masking', 'token')
    p_mask = kwargs.get('p_mask', 0.15)
    transform = transforms.Compose(
            [
                MSATokenize(rna2index),
                RandomMSAMasking(p=p_mask, mode=masking, mask_token=rna2index['MASK_TOKEN']),
                ExplicitPositionalEncoding(),
            ])
    head = InpaintingHead(dim, len(rna2index)-num_special_tokens)

    return transform, {'inpainting': head}, {'inpainting': CrossEntropyLoss()}
