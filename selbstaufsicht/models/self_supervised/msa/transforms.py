from functools import partial
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
    def __call__(self, x):
        x['msa'] = torch.tensor([[self.mapping[letter] for letter in sequence] for sequence in x['msa']], dtype=torch.long)
        if 'START_TOKEN' in self.mapping:
            prefix = torch.full((x['msa'].size(0), 1), self.mapping['START_TOKEN'], dtype=torch.int)
            x['msa'] = torch.cat([prefix, x['msa']], dim=-1)
        if 'contrastive' in x:
            x['contrastive'] = torch.tensor([[self.mapping[letter] for letter in sequence] for sequence in x['contrastive']], dtype=torch.long)
            prefix = torch.full((x['contrastive'].size(0), 1), self.mapping['START_TOKEN'], dtype=torch.int)
            x['contrastive'] = torch.cat([prefix, x['contrastive']], dim=-1)

        return x


class RandomMSACropping():
    def __init__(self, length, contrastive=False):
        self.length = length
        self.contrastive = contrastive

    def __call__(self, x):
        alen = x['msa'].get_alignment_length()
        if alen <= self.length:
            return x
        start = torch.randint(alen - self.length, (1,)).item()
        result = {'msa': x['msa'][:, start:start + self.length]}
        if self.contrastive:
            start = torch.randint(alen - self.length, (1,)).item()
            contrastive_x = x['msa'][:, start:start + self.length]
            result['contrastive'] = contrastive_x
        return result


class RandomMSAMasking():
    # TODO other tokens instead of mask token
    def __init__(self, p, mode, mask_token, contrastive=False, start_token=True):
        self.p = p
        self.mask_token = mask_token
        self.masking_fn = _get_masking_fn(mode, start_token)
        self.contrastive = contrastive

    def __call__(self, x):
        y = {}
        if type(x) == tuple:
            x, y = x
        target = None
        if type(x) == dict:
            masked_msa, mask, target = self.masking_fn(x['msa'], self.p, self.mask_token)
            x['msa'] = masked_msa
            x['mask'] = mask
        elif type(x) == torch.Tensor:
            masked_msa, mask, target = self.masking_fn(x, self.p, self.mask_token)
            x = {'msa': masked_msa, 'mask': mask}
        else:
            raise ValueError()
        y['inpainting'] = target

        if self.contrastive:
            contrastive_x, _, _ = self.masking_fn(x['contrastive'], self.p, self.mask_token)
            x['contrastive'] = contrastive_x
        return x, y


class RandomMSAShuffling():
    def __init__(self, permutations=None, minleader=1, mintrailer=0, delimiter_token=None, num_partitions=None, num_classes=None, contrastive=False):
        if permutations is None and (num_partitions is None or num_classes is None):
            raise ValueError("Permutations have to be given explicitely or parameters to generate them.")

        # TODO always include 'no transformation'
        if permutations is None:
            perm_indices = list(range(math.factorial(num_partitions)))
            random.shuffle(perm_indices)
            self.permutations = torch.stack([lehmer_encode(i, num_partitions) for i in perm_indices[:num_classes]]).unsqueeze(0)
        else:
            # NOTE attribute permutations is expected to have shape [num_classes, num_partitions]
            # NOTE add singleton dim for later expansion to num_seq dim
            self.permutations = permutations.unsqueeze(0)
        self.num_partitions = max(self.permutations[0, 0])
        self.num_classes = len(self.permutations[0])
        self.minleader = minleader
        self.mintrailer = mintrailer
        self.delimiter_token = delimiter_token
        self.contrastive = contrastive

    def __call__(self, x, label=None):
        '''
        x is either a tensor or a tuple of input and target dictionaries
        '''
        y = dict()
        if type(x) == tuple:
            x, y = x
        if type(x) == dict:
            num_seq = x['msa'].size(0)
            if label is None:
                label = torch.randint(0, self.num_classes, (num_seq,))
            shuffled_msa = _jigsaw(x['msa'],
                                   self.permutations.expand(num_seq, -1, -1)[range(num_seq),
                                   label], delimiter_token=self.delimiter_token,
                                   minleader=self.minleader,
                                   mintrailer=self.mintrailer)
            x['msa'] = shuffled_msa
        else:
            num_seq = x.size(0)
            if label is None:
                label = torch.randint(0, self.num_classes, (num_seq,))
            shuffled_msa = _jigsaw(x,
                                   self.permutations.expand(num_seq, -1, -1)[range(num_seq), label],
                                   delimiter_token=self.delimiter_token,
                                   minleader=self.minleader,
                                   mintrailer=self.mintrailer)
            x = {'msa': shuffled_msa}
        y['jigsaw'] = label
        if self.contrastive:
            contrastive_perm = torch.randint(0, self.num_classes, (num_seq,))
            contrastive_x = _jigsaw(x['contrastive'],
                                    self.permutations.expand(num_seq, -1, -1)[range(num_seq),
                                    contrastive_perm],
                                    delimiter_token=self.delimiter_token,
                                    minleader=self.minleader,
                                    mintrailer=self.mintrailer)
            x['contrastive'] = contrastive_x

        return x, y


class RandomMSASubsampling():
    def __init__(self, num_sequences, contrastive=False, mode='uniform'):
        self.contrastive = contrastive
        self.sampling_fn = _get_msa_subsampling_fn(mode)
        self.nseqs = num_sequences

    def __call__(self, x):
        if self.contrastive:
            return {'msa': self.sampling_fn(x, self.nseqs), 'contrastive': self.sampling_fn(x, self.nseqs)}
        return {'msa': self.sampling_fn(x, self.nseqs)}


class ExplicitPositionalEncoding():
    def __init__(self, axis=-1, abs_factor=1000):
        self.axis = axis
        self.abs_factor = abs_factor
        

    def __call__(self, x):
        if type(x) == tuple:
            x, target = x
        else:
            # TODO this is a contrastive dummy label, the model replaces it with the embedding of the contrastive input, maybe it would be better to not have this be needed?
            target = {'contrastive': torch.tensor(0)}
            assert type(x) == dict

        msa = x['msa']
        size = msa.size(self.axis)
        absolute = torch.arange(0, size, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
        relative = absolute / size
        absolute = absolute / self.abs_factor
        if 'aux_features' not in x:
            x['aux_features'] = torch.cat((absolute, relative), dim=-1)
        else:
            x['aux_features'] = torch.cat((msa['aux_features'], absolute, relative), dim=-1)
        
        if 'contrastive' in x:
            msa = x['contrastive']
            size = msa.size(self.axis)
            absolute = torch.arange(0, size, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
            relative = absolute / size
            absolute = absolute / self.abs_factor
            if 'aux_features_contrastive' not in x:
                x['aux_features_contrastive'] = torch.cat((absolute, relative), dim=-1)
            else:
                x['aux_features_contrastive'] = torch.cat((msa['aux_features_contrastive'], absolute, relative), dim=-1)

        return x, target


# TODO test
# TODO maybe remove possible shortcut of e.g.
# >AAA|BB
# >BB|AAA
# should the leader and trailer be adapted such, that all partitions are of the same size?

def _jigsaw(msa, permutations, delimiter_token=None, minleader=1, mintrailer=0):
    '''
    msa is tensor of size (num_seqs, seq_len)
    '''
    # TODO relative leader and trailer?
    # TODO minimum partition size?
    # TODO optimize
    nres = msa.size(-1)
    assert permutations.size(0) == msa.size(0)
    npartitions = permutations.size(-1)
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

    lines = []
    for (i, permutation) in enumerate(permutations):
        line_chunks = []
        for p in permutation:
            if delimiter_token is not None:
                line_chunks.append(torch.full((1,), delimiter_token, dtype=torch.int))
            line_chunks.append(partitions[p.item()][i])
        if delimiter_token is not None:
            line_chunks.append(torch.full((1,), delimiter_token, dtype=torch.int))
        lines.append(torch.cat(line_chunks, dim=0).unsqueeze(0))

    chunks.append(torch.cat(lines, dim=0))
    chunks.append(trailer)

    jigsawed_msa = torch.cat(chunks, dim=-1)

    return jigsawed_msa


# TODO add capabilities for not masking and replace with random other token
def _block_mask_msa(msa, p, mask_token, start_token=True):
    """
    masks out a contiguous block of columns in the given msa
    """

    total_length = msa.size(-1) - int(start_token)
    mask_length = int(total_length * p)
    begin = torch.randint(total_length - mask_length, (1, )).item() + int(start_token)
    end = begin + mask_length

    mask = torch.zeros_like(msa, dtype=torch.bool)
    mask[:, begin:end] = True

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _column_mask_msa_indexed(msa, col_indices, mask_token, start_token=True):
    """
    masks out a given set of columns in the given msa
    """

    mask = torch.zeros_like(msa, dtype=torch.bool)
    mask[:, col_indices + int(start_token)] = True

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _column_mask_msa(msa, p, mask_token, start_token=True):
    """
    masks out a random set of columns in the given msa
    """

    col_num = msa.size(-1) - int(start_token)
    col_indices = torch.arange(col_num, dtype=torch.long)
    col_mask = torch.full((col_num,), p)
    col_mask = torch.bernoulli(col_mask).to(torch.bool)
    masked_col_indices = (col_mask * col_indices)
    return _column_mask_msa_indexed(msa, masked_col_indices, mask_token, start_token=start_token)


# TODO should seq start token be excluded from masking?
def _token_mask_msa(msa, p, mask_token, start_token=True):
    """
    masks out random tokens uniformly sampled from the given msa
    """

    mask = torch.full(msa.size(), p)
    mask[:, :int(start_token)] = 0.
    mask = torch.bernoulli(mask).to(torch.bool)

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _get_masking_fn(mode, start_token):
    if mode == 'token':
        return partial(_token_mask_msa, start_token=start_token)
    elif mode == 'column':
        return partial(_column_mask_msa, start_token=start_token)
    elif mode == 'block':
        return partial(_block_mask_msa, start_token=start_token)
    raise ValueError('unknown token masking mode', mode)


def _subsample_uniform(msa, nseqs):
    max_nseqs = len(msa)
    if max_nseqs > nseqs:
        indices = torch.randperm(max_nseqs)[:nseqs]
        msa = MultipleSeqAlignment([msa[i.item()] for i in indices])
    return msa


def _hamming_distance(seq_1, seq_2):
    assert len(seq_1) == len(seq_2), "Both sequences are required to have the same length!"
    return sum(n_1 != n_2 for n_1, n_2 in zip(seq_1, seq_2))


def _hamming_distance_matrix(msa):
    hd_matrix = torch.zeros((len(msa), len(msa)), dtype=torch.int64)
    
    # computes upper triangular part, without diagonal
    for idx_1, seq_1 in enumerate(msa):
        for idx_2, seq_2 in enumerate(msa):
            if idx_2 <= idx_1:
                continue
            hd_matrix[idx_1, idx_2] = _hamming_distance(seq_1.seq, seq_2.seq)
    
    # make matrix symmetric for easier handling
    return hd_matrix + hd_matrix.T


def _maximize_diversity_naive(msa, msa_indices, nseqs, sampled_msa):
    # naive strategy: compute hamming distances on-the-fly, when needed
    hd_matrix = torch.zeros((len(msa_indices), len(sampled_msa)), dtype=torch.int64)
        
    for idx_1, idx_msa in enumerate(msa_indices):
        seq_1 = msa[idx_msa]
        for idx_2, seq_2 in enumerate(sampled_msa):
            hd_matrix[idx_1, idx_2] = _hamming_distance(seq_1.seq, seq_2.seq)
    
    # average over already sampled sequences
    avg_hd = torch.mean(hd_matrix.float(), dim=1)
    # find sequence with maximum average hamming distance
    idx_max = torch.argmax(avg_hd).item()
    idx_msa_max = msa_indices[idx_max]
        
    sampled_msa.append(msa[idx_msa_max])
    msa_indices.pop(idx_max)
    nseqs -= 1
    
    if nseqs == 0:
        return sampled_msa
    else:
        return _maximize_diversity_naive(msa, msa_indices, nseqs, sampled_msa)
                

def _maximize_diversity_cached(msa, msa_indices, nseqs, sampled_msa, sampled_msa_indices, hd_matrix):
    # cached strategy: use pre-computed hamming distances
    indices = tuple(zip(*[(msa_idx, sampled_msa_idx) for msa_idx in msa_indices for sampled_msa_idx in sampled_msa_indices]))
    hd_matrix_reduced = hd_matrix[indices[0], indices[1]].view(len(msa_indices), len(sampled_msa_indices))
    
    # average over already sampled sequences
    if hd_matrix_reduced.dim() == 2:
        avg_hd = torch.mean(hd_matrix_reduced.float(), dim=1)
    else:
        avg_hd = hd_matrix_reduced.float()
    # find sequence with maximum average hamming distance
    idx_max = torch.argmax(avg_hd).item()
    idx_msa_max = msa_indices[idx_max]
    
    sampled_msa.append(msa[idx_msa_max])
    msa_indices.pop(idx_max)
    sampled_msa_indices.append(idx_msa_max)
    nseqs -= 1
    
    if nseqs == 0:
        return sampled_msa
    else:
        return _maximize_diversity_cached(msa, msa_indices, nseqs, sampled_msa, sampled_msa_indices, hd_matrix)
    

def _subsample_diversity_maximizing(msa, nseqs, contrastive=False):
    # since the function is deterministic and contrastive input should be different from the regular input, it is sampled randomly
    if contrastive:
        return _subsample_uniform(msa, nseqs)
    
    # depending on the total number of sequences and the number of sequences to be subsampled, choose computation strategy:
    # either compute and cache all hamming distanced between distinct sequences beforehand or use the naive implementation with potentially repeating comparisons
    
    n = len(msa)
    # exclude reference seq
    m = min(nseqs, n) - 1
    
    # symmetric, reflexive n:n relation
    comparisons_cached = (n**2 - n) / 2
    # equivalent to sum_{i=1}^{m} i*(N-i), which is the cumulative number of comparisons after the m-th recursive call
    comparisons_naive = n * m * (m + 1) / 2 - m * (m + 1) * (2 * m + 1) / 6
    
    if comparisons_cached <= comparisons_naive:
        hd_matrix = _hamming_distance_matrix(msa)
        return _maximize_diversity_cached(msa, list(range(1, n)), m, msa[0:1], [0], hd_matrix)
    else:
        return _maximize_diversity_naive(msa, list(range(1, n)), m, msa[0:1])


def _get_msa_subsampling_fn(mode):
    if mode == 'uniform':
        return _subsample_uniform
    if mode == 'diversity':
        return _subsample_diversity_maximizing
    raise ValueError('unkown msa sampling mode', mode)
