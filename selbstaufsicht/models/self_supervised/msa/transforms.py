import numpy as np
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


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


# TODO
class RandomMSAColumnMasking(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, msa_tensor):
        raise
        return


# TODO
class RandomMSATokenMasking(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, msa_tensor):
        raise
        return


class MSAJigsawTransform(object):
    def __init__(self, npartitions=2, minfringe=5, ):
        return


class MSAContrastiveTransform(object):
    def __init__(self):
        return


class MSA2Tensor(object):
    def __init__(self, mapping, device=None):
        self.mapping = mapping
        self.device = device
        return

    def __call__(self, msa):
        return torch.tensor([[self.mapping[letter] for letter in sequence] for sequence in msa], device=self.device)


class MSA2array(object):
    '''
    takes an MSA object and produces a numpy char array
    '''
    def __init__(self):
        return

    def __call__(self, msa):
        return np.array([list(rec) for rec in msa], dtype='<U1')


class OneHotMSAArray(object):
    def __init__(self, mapping):
        maxind = 256

        self.mapping = np.full((maxind, ), -1)
        for k in mapping:
            self.mapping[ord(k)] = mapping[k]

    def __call__(self, msa_array):
        """
        msa_array: byte array
        """

        return self.mapping[msa_array.view(np.uint32)]


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


# TODO adapt to tensorized input
def _block_mask_msa(msa, begin, end, mask_token='*'):
    masked = msa[:, begin:end]
    mask = MultipleSeqAlignment([SeqRecord(Seq(mask_token * (end - begin)), id=r.id) for r in msa])
    msa = msa[:, :begin] + mask + msa[:, end:]
    return msa, masked


# TODO test this, not sure this kind of indexing works without casting to numpy array
def _column_mask_msa(msa_tensor, col_indices, mask_token='*'):
    masked = msa_tensor[col_indices]
    msa_tensor[col_indices] = mask_token
    return msa_tensor, masked


# TODO test
def _token_mask_msa(msa_tensor, coords, mask_token_index):
    masked = msa_tensor[coords]
    msa_tensor[coords] = mask_token_index
    return msa_tensor, masked


def _subsample(msa, contrastive=False):
    return
