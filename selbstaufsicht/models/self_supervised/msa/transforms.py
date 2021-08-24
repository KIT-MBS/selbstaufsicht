import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


class MSATokenize():
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, msa):
        torch.tensor([[self.mapping[letter] for letter in sequence] for sequence in msa], dtype=torch.float)


class RandomMSAMasking():
    def __init__(self, p, mode, mask_token):
        self.p = p
        self.mask_token = mask_token
        self.masking_fn = _get_masking_fn(mode)

    def __call__(self, x):
        return self.masking_fn(x, self.p, self.mask_token)


class ExplicitPositionalEncoding():
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        size = x.size(self.axis)
        absolute = torch.arange(0, size, dtype=torch.float)
        relative = absolute/size

        return {'msa': x, 'sequential_features': torch.cat((absolute, relative), dim=self.axis)}


# class OneHotMSAArray(object):
#     def __init__(self, mapping):
#         maxind = 256
#
#         self.mapping = np.full((maxind, ), -1)
#         for k in mapping:
#             self.mapping[ord(k)] = mapping[k]
#
#     def __call__(self, msa_array):
#         """
#         msa_array: byte array
#         """
#
#         return self.mapping[msa_array.view(np.uint32)]


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
def _block_mask_msa(msa, p, mask_token='*'):
    """
    masks out a contiguous block of columns in the given msa
    """
    total_length = msa.size(-1)
    mask_length = int(total_length * p)
    begin = torch.randint(total_length-mask_length, (1)).item()
    end = begin + mask_length
    masked = msa[:, begin:end]
    mask = MultipleSeqAlignment([SeqRecord(Seq(mask_token * (end - begin)), id=r.id) for r in msa])
    msa = msa[:, :begin] + mask + msa[:, end:]
    return msa, masked


# TODO test this, not sure this kind of indexing works without casting to numpy array
def _column_mask_msa(msa_tensor, col_indices, mask_token='*'):
    """
    masks out a random set of columns in the given msa
    """
    masked = msa_tensor[col_indices]
    msa_tensor[col_indices] = mask_token
    return msa_tensor, masked


# TODO test
def _token_mask_msa(msa_tensor, coords, mask_token_index):
    """
    masks out random tokens uniformly sampled from the given msa
    """
    masked = msa_tensor[coords]
    msa_tensor[coords] = mask_token_index
    return msa_tensor, masked


def _get_masking_fn(mode):
    if mode == 'token':
        return _token_mask_msa
    elif mode == 'column':
        return _column_mask_msa
    elif mode == 'block':
        return _block_mask_msa
    raise ValueError('unknown token masking mode', mode)


def _subsample_uniform(msa, contrastive=False):
    # TODO
    raise


def _get_msa_subsampling_fn(mode):
    raise


def _subsample_diversity_maximizing(msa, contrastive=False):
    # TODO
    raise
