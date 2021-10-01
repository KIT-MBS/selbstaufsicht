import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

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


class RandomMSAMasking():
    def __init__(self, p, mode, mask_token):
        self.p = p
        self.mask_token = mask_token
        self.masking_fn = _get_masking_fn(mode)

    def __call__(self, x):
        masked_msa, mask, target = self.masking_fn(x, self.p, self.mask_token)
        # TODO generalize for other task orders
        x = {'msa': masked_msa, 'mask': mask}
        y = {'inpainting': target}
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


# TODO do this before cropping?
class ExplicitPositionalEncoding():
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        x, target = x

        msa = x['msa']
        size = msa.size(self.axis)
        absolute = torch.arange(0, size, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        relative = absolute/size
        if 'aux_features' not in x:
            x['aux_features'] = torch.cat((absolute, relative), dim=0)
        else:
            x['aux_features'] = torch.cat((msa['aux_features'], absolute, relative), dim=0)

        return x, target


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
# TODO add capabilities for not masking and replace with random other token
def _block_mask_msa(msa, p, mask_token):
    """
    masks out a contiguous block of columns in the given msa
    """
    # TODO
    raise
    total_length = msa.size(-1)
    mask_length = int(total_length * p)
    begin = torch.randint(total_length-mask_length, (1)).item()
    end = begin + mask_length
    masked = msa[:, begin:end]
    mask = MultipleSeqAlignment([SeqRecord(Seq(mask_token * (end - begin)), id=r.id) for r in msa])
    msa = msa[:, :begin] + mask + msa[:, end:]
    return msa, mask, masked


# TODO test this, not sure this kind of indexing works without casting to numpy array
def _column_mask_msa(msa, col_indices, mask_token):
    """
    masks out a random set of columns in the given msa
    """
    raise
    mask = None
    masked = msa[col_indices]
    msa[col_indices] = mask_token
    return msa, mask, masked


# TODO test
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
