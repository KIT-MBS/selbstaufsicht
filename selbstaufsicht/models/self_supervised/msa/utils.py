from torch.nn import CrossEntropyLoss
from torch.nn import ModuleDict
from selbstaufsicht import transforms
from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, RandomMSAMasking, ExplicitPositionalEncoding
from selbstaufsicht.models.self_supervised.msa.transforms import RandomMSACropping, RandomMSASubsampling, RandomMSAShuffling
from .modules import InpaintingHead, JigsawHead
from torchmetrics import Accuracy

# NOTE mask and delimiter tokens can not be reconstructed


def get_tasks(tasks, dim, **kwargs):
    # TODO other tasks
    if not set(tasks) <= {'inpainting', 'jigsaw'}:
        raise NotImplementedError('only inpainting or jigsaw')
    if 'masking' in kwargs and kwargs['masking'] != 'token':
        raise NotImplementedError('only token masking')

    transformslist = [
        RandomMSASubsampling(5),
        RandomMSACropping(50),
        MSATokenize(rna2index)]

    if 'inpainting' in tasks:
        masking = kwargs.get('masking', 'token')
        p_mask = kwargs.get('p_mask', 0.15)
        transformslist.append(RandomMSAMasking(p=p_mask, mode=masking, mask_token=rna2index['MASK_TOKEN']))

    if 'jigsaw' in tasks:
        jigsaw_classes = 10
        transformslist.append(RandomMSAShuffling(delimiter_token=rna2index['DELIMITER_TOKEN'], num_partitions=5, num_classes=jigsaw_classes))

    transformslist.append(ExplicitPositionalEncoding())

    transform = transforms.Compose(transformslist)

    metrics = ModuleDict()

    task_heads = ModuleDict()
    task_losses = dict()
    if 'inpainting' in tasks:
        head = InpaintingHead(dim, len(rna2index) - 1)  # NOTE never predict mask token
        task_heads['inpainting'] = head

        task_losses['inpainting'] = CrossEntropyLoss()
        metrics['inpainting'] = ModuleDict({'acc': Accuracy()})

    # TODO always include 'no transformation'
    if 'jigsaw' in tasks:
        if 'jigsaw_classes' in kwargs:
            raise
        head = JigsawHead(dim, jigsaw_classes)
        task_heads['jigsaw'] = head
        task_losses['jigsaw'] = CrossEntropyLoss()
        metrics['jigsaw'] = ModuleDict({'acc': Accuracy()})

    return transform, task_heads, task_losses, metrics


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

        pass
        # xs = [sample[0] for sample in x]
        # ys = [sample[1] for sample in x]

        # out_xs = {}
        # out_ys = {}

        # return out_xs, out_ys

    def _pad_collate(x):
        pass

    def _flat_collate(x):
        pass
