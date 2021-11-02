import torch
from Bio.Align import MultipleSeqAlignment
import math
import random

from selbstaufsicht.utils import lehmer_encode

# TODO task scheduling a la http://bmvc2018.org/contents/papers/0345.pdf


class MSATokenize():
    def __init__(self, mapping):
        self.mapping = mapping

    # TODO maybe do tensor mapping instead of dict as in:
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
    def __call__(self, msa):
        return torch.tensor([[self.mapping[letter] for letter in sequence] for sequence in msa], dtype=torch.long)


class RandomMSACropping():
    def __init__(self, length):
        self.length = length

    def __call__(self, x):
        # TODO interface with subsampling
        alen = x.get_alignment_length()
        if alen <= self.length:
            return x
        start = torch.randint(alen - self.length, (1,)).item()
        return x[:, start:start + self.length]


class RandomMSAMasking():
    # TODO other tokens instead of mask token
    def __init__(self, p, mode, mask_token):
        self.p = p
        self.mask_token = mask_token
        self.masking_fn = _get_masking_fn(mode)

    def __call__(self, x):
        masked_msa, mask, target = self.masking_fn(x, self.p, self.mask_token)
        # TODO check task order
        x = {'msa': masked_msa, 'mask': mask}
        y = {'inpainting': target}
        return x, y


class RandomMSAShuffling():
    def __init__(self, permutations=None, minleader=0, mintrailer=0, delimiter_token=None, num_partitions=None, num_classes=None):
        if permutations is None and (num_partitions is None or num_classes is None):
            raise ValueError("Permutations have to be given explicitely or parameters to generate them.")
        self.permutations = permutations

        if permutations is None:
            perm_indices = list(range(math.factorial(num_partitions)))
            random.shuffle(perm_indices)
            self.permutations = [lehmer_encode(i, num_partitions) for i in perm_indices[:num_classes]]
        self.num_partitions = max(self.permutations[0])
        self.num_classes = len(self.permutations)
        self.minleader = minleader
        self.mintrailer = mintrailer
        self.delimiter_token = delimiter_token

    def __call__(self, x):
        '''
        x is either a tensor or a tuple of input and target dictionaries
        '''
        y = dict()
        if type(x) == tuple:
            x, y = x
        label = torch.randint(0, self.num_classes, (1,)).item()
        if type(x) == dict:
            shuffled_msa = _jigsaw(x['msa'], self.permutations[label], delimiter_token=self.delimiter_token, minleader=self.minleader, mintrailer=self.mintrailer)
            x['msa'] = shuffled_msa
        else:
            shuffled_msa = _jigsaw(x, self.permutations[label], delimiter_token=self.delimiter_token, minleader=self.minleader, mintrailer=self.mintrailer)
            x = {'msa': shuffled_msa}
        y['jigsaw'] = label

        return x, y


class RandomMSASubsampling():
    def __init__(self, num_sequences, contrastive=False, mode='uniform'):
        if contrastive:
            raise
        self.contrastive = contrastive
        self.sampling_fn = _get_msa_subsampling_fn(mode)
        self.nseqs = num_sequences

    def __call__(self, x):
        return self.sampling_fn(x, self.nseqs, self.contrastive)


class ExplicitPositionalEncoding():
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        x, target = x

        msa = x['msa']
        size = msa.size(self.axis)
        absolute = torch.arange(0, size, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
        relative = absolute / size
        if 'aux_features' not in x:
            x['aux_features'] = torch.cat((absolute, relative), dim=-1)
        else:
            x['aux_features'] = torch.cat((msa['aux_features'], absolute, relative), dim=-1)

        return x, target


def _jigsaw(msa, permutation, delimiter_token, minleader=0, mintrailer=0):
    '''
    msa is tensor of size (num_seqs, seq_len)
    '''
    # TODO relative leader and trailer?
    # TODO minimum partition size?
    # TODO optimize
    nres = msa.size(-1)
    nseqs = msa.size(0)
    npartitions = len(permutation)
    partition_length = (nres - minleader - mintrailer) // npartitions
    core_leftover = nres - minleader - mintrailer - (partition_length * npartitions)
    if minleader < minleader + core_leftover:
        offset = torch.randint(minleader, minleader + core_leftover, (1,)).item()
    else:
        offset = 0

    leader = msa[:, :offset]
    trailer = msa[:, offset + npartitions * partition_length:]
    partitions = [msa[:, offset + i * partition_length: offset + (i + 1) * partition_length] for i in range(npartitions)]

    chunks = list()
    if leader.numel() > 0:
        chunks = [leader]

    for p in permutation:
        if delimiter_token is not None:
            chunks.append(torch.full((nseqs, 1), delimiter_token, dtype=torch.int))
        chunks.append(partitions[p])
    if delimiter_token is not None:
        chunks.append(torch.full((nseqs, 1), delimiter_token, dtype=torch.int))
    chunks.append(trailer)

    jigsawed_msa = torch.cat(chunks, dim=-1)

    return jigsawed_msa


# TODO add capabilities for not masking and replace with random other token
def _block_mask_msa(msa, p, mask_token):
    """
    masks out a contiguous block of columns in the given msa
    """

    total_length = msa.size(-1)
    mask_length = int(total_length * p)
    begin = torch.randint(total_length - mask_length, (1, )).item()
    end = begin + mask_length

    mask = torch.zeros_like(msa, dtype=torch.bool)
    mask[:, begin:end] = True

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _column_mask_msa_indexed(msa, col_indices, mask_token):
    """
    masks out a given set of columns in the given msa
    """

    mask = torch.zeros_like(msa, dtype=torch.bool)
    mask[:, col_indices] = True

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _column_mask_msa(msa, p, mask_token):
    """
    masks out a random set of columns in the given msa
    """

    col_num = msa.size(-1)
    col_indices = torch.arange(col_num, dtype=torch.long)
    col_mask = torch.full((col_num,), p)
    col_mask = torch.bernoulli(col_mask).to(torch.bool)
    masked_col_indices = (col_mask * col_indices)
    return _column_mask_msa_indexed(msa, masked_col_indices, mask_token)


def _token_mask_msa(msa, p, mask_token):
    """
    masks out random tokens uniformly sampled from the given msa
    """

    mask = torch.full(msa.size(), p)
    mask = torch.bernoulli(mask).to(torch.bool)

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _get_masking_fn(mode):
    if mode == 'token':
        return _token_mask_msa
    elif mode == 'column':
        return _column_mask_msa
    elif mode == 'block':
        return _block_mask_msa
    raise ValueError('unknown token masking mode', mode)


def _subsample_uniform(msa, nseqs, contrastive=False):
    max_nseqs = len(msa)
    if max_nseqs > nseqs:
        indices = torch.randperm(max_nseqs)[:nseqs]
        msa = MultipleSeqAlignment([msa[i.item()] for i in indices])
    return msa


def _subsample_diversity_maximizing(msa, nseqs, contrastive=False):
    # TODO
    raise


def _get_msa_subsampling_fn(mode):
    if mode == 'uniform':
        return _subsample_uniform
    if mode == 'diversity':
        return _subsample_diversity_maximizing
    raise ValueError('unkown msa sampling mode', mode)
