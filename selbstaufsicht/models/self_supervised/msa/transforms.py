import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

# from ...utils import rna_to_index


class MSADataTrainTransform(object):
    """
    Transforms for multiple sequence alignment data

    Transform::
        String of RNA letters to token indices
    """

    def __init__(self, masking=True, jigsaw=True, contrastive=True, ambiguity=None):
        return

    def __call__(self, msa):
        # return [[rna_to_index[a] for a in sequence] for sequence in msa]
        pass

        # return (sample, target)


class MSAMaskingTransform(object):
    def __init__(self):
        return


class MSAJigsawTransform(object):
    def __init__(self, npartitions=2, minfringe=5, ):
        return


class MSAContrastiveTransform(object):
    def __init__(self):
        return


def _jigsaw(msa, permutation, minleader=0, mintrailer=0, delimiter_token='|'):
    # TODO relative leader and trailer?
    # TODO minimum partition size?
    nres = msa.get_alignment_length()
    npartitions = len(permutation)
    partition_length = (nres - minleader - mintrailer) // npartitions
    core_leftover = nres - minleader - mintrailer - (partition_length * npartitions)
    offset = torch.randint(minleader, minleader + core_leftover, (1)).item()

    leader = msa[:, :offset]
    trailer = msa[:, offset + npartitions * partition_length:]
    partitions = [msa[offset + i * partition_length: offset + (i + 1) * partition_length] for i in range(npartitions)]

    jigsawed_msa = leader
    for p in partitions:
        jigsawed_msa += MultipleSeqAlignment([SeqRecord(Seq(delimiter_token), id=r.id) for r in msa])
        jigsawed_msa += partitions
    jigsawed_msa += MultipleSeqAlignment([SeqRecord(Seq(delimiter_token), id=r.id) for r in msa])
    jigsawed_msa += trailer

    return jigsawed_msa


def _mask_msa(msa, begin, end, mask_token='*'):
    masked = msa[:, begin:end]
    mask = MultipleSeqAlignment([SeqRecord(Seq(mask_token * (end - begin)), id=r.id) for r in msa])
    msa = msa[:, :begin] + mask + msa[:, end:]
    return msa, masked


def _subsample(msa, contrastive=False):
    return
