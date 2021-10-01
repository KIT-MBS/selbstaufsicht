from torch.nn import CrossEntropyLoss
from torch.nn import ModuleDict
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding, RandomMSACropping, RandomMSASubsampling
from .modules import InpaintingHead
from torchmetrics import Accuracy

# NOTE mask and delimiter tokens can not be reconstructed


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
            RandomMSASubsampling(5),
            RandomMSACropping(50),
            MSATokenize(rna2index),
            RandomMSAMasking(p=p_mask, mode=masking, mask_token=rna2index['MASK_TOKEN']),
            ExplicitPositionalEncoding(),
        ])
    head = InpaintingHead(dim, len(rna2index) - 1)  # NOTE never predict mask token
    metrics = ModuleDict({'inpainting': ModuleDict({'acc': Accuracy()})})

    return transform, ModuleDict({'inpainting': head}), {'inpainting': CrossEntropyLoss()}, metrics


class MSACollator():
    def __init__(self):
        pass

    def __call__(self, x):
        """
        x: list of tuples of dicts: e.g. [({input1}, {target1}), ({input2}, {target2})]
        input contains: {'msa': tensor, optional 'mask': tensor, 'aux_features': tensor}
        target contains one or more of {'inpainting': 1dtensor, 'jigsaw': int, 'contrastive': tensor}
        """
        # TODO msas, masks, and aux_features have to be padded to fit and stacked
        # TODO inpainting and contrastive targets are cated to a 1d tensor,  jigsaw has to be stacked

        xs = [sample[0] for sample in x]
        ys = [sample[1] for sample in x]

        out_xs = {}
        out_ys = {}

        return out_xs, out_ys

    def _pad_collate(x):
        pass

    def _flat_collate(x):
        pass
