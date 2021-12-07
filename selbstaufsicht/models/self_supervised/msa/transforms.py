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

    def __call__(self, x, y):
        x['msa'] = torch.tensor([[self.mapping[letter] for letter in sequence] for sequence in x['msa']], dtype=torch.long)
        if 'contrastive' in x:
            x['contrastive'] = torch.tensor([[self.mapping[letter] for letter in sequence] for sequence in x['contrastive']], dtype=torch.long)

        if 'START_TOKEN' in self.mapping:
            prefix = torch.full((x['msa'].size(0), 1), self.mapping['START_TOKEN'], dtype=torch.int)
            x['msa'] = torch.cat([prefix, x['msa']], dim=-1)
            if 'contrastive' in x:
                prefix = torch.full((x['contrastive'].size(0), 1), self.mapping['START_TOKEN'], dtype=torch.int)
                x['contrastive'] = torch.cat([prefix, x['contrastive']], dim=-1)

        return x, y


class RandomMSACropping():
    def __init__(self, length, contrastive=False):
        self.length = length
        self.contrastive = contrastive

    def __call__(self, x, y):
        msa = x['msa'][:, :]
        if x['msa'].get_alignment_length() > self.length:
            start = torch.randint(msa.get_alignment_length() - self.length, (1,)).item()
            x['msa'] = x['msa'][:, start:start + self.length]

        if self.contrastive:
            contrastive_msa = x.get('contrastive', msa)
            if contrastive_msa.get_alignment_length() > self.length:
                start = torch.randint(contrastive_msa.get_alignment_length() - self.length, (1,)).item()
                contrastive_msa = contrastive_msa[:, start:start + self.length]
            x['contrastive'] = contrastive_msa
        return x, y


class RandomMSAMasking():
    # TODO other tokens instead of mask token
    def __init__(self, p, mode, mask_token, contrastive=False, start_token=True):
        self.p = p
        self.mask_token = mask_token
        self.masking_fn = _get_masking_fn(mode, start_token)
        self.contrastive = contrastive

    def __call__(self, x, y):
        masked_msa, mask, target = self.masking_fn(x['msa'], self.p, self.mask_token)
        x['msa'] = masked_msa
        x['mask'] = mask
        y['inpainting'] = target
        if self.contrastive:
            x['contrastive'], _, _ = self.masking_fn(x['contrastive'], self.p, self.mask_token)
        return x, y


class RandomMSAShuffling():
    def __init__(self, permutations=None, minleader=1, mintrailer=0, delimiter_token=None, num_partitions=None, num_classes=None, contrastive=False):
        if permutations is None and (num_partitions is None or num_classes is None):
            raise ValueError("Permutations have to be given explicitely or parameters to generate them.")

        if permutations is None:
            perm_indices = list(range(1, math.factorial(num_partitions)))
            random.shuffle(perm_indices)
            # NOTE always include 'no transformation'
            perm_indices.insert(0, 0)
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

    def __call__(self, x, y, label=None):
        num_seq = x['msa'].size(0)
        if label is None:
            label = torch.randint(0, self.num_classes, (num_seq,))
        shuffled_msa = _jigsaw(x['msa'],
                               self.permutations.expand(num_seq, -1, -1)[range(num_seq),
                               label], delimiter_token=self.delimiter_token,
                               minleader=self.minleader,
                               mintrailer=self.mintrailer)
        x['msa'] = shuffled_msa
        y['jigsaw'] = label
        if self.contrastive:
            contrastive_perm = torch.randint(0, self.num_classes, (num_seq,))
            x['contrastive'] = _jigsaw(x['contrastive'],
                                       self.permutations.expand(num_seq, -1, -1)[range(num_seq),
                                       contrastive_perm],
                                       delimiter_token=self.delimiter_token,
                                       minleader=self.minleader,
                                       mintrailer=self.mintrailer)

        return x, y


class RandomMSASubsampling():
    def __init__(self, num_sequences, contrastive=False, mode='uniform'):
        self.contrastive = contrastive
        self.sampling_fn = _get_msa_subsampling_fn(mode)
        self.nseqs = num_sequences

    def __call__(self, x, y):
        msa = x['msa'][:, :]
        x['msa'] = self.sampling_fn(msa, self.nseqs)
        if self.contrastive:
            x['contrastive'] = self.sampling_fn(msa, self.nseqs)
        return x, y


# TODO this is sort of a hacky way to fix the incorrect way of PE without having to touch everything
class ExplicitPositionalEncoding():
    def __init__(self, max_seqlen=5000):
        self.max_seqlen = max_seqlen

    def __call__(self, x, y):
        msa = x['msa']
        seqlen = msa.size(-1)
        if seqlen > self.max_seqlen:
            raise ValueError(f'Sequence dimension in input too large: {seqlen} > {self.max_seqlen}')
        absolute = torch.arange(1, seqlen + 1, dtype=torch.long).unsqueeze(0)
        if 'aux_features' not in x:
            x['aux_features'] = absolute
        else:
            raise

        if 'contrastive' in x:
            msa = x['contrastive']
            seqlen = msa.size(-1)

            absolute = torch.arange(1, seqlen + 1, dtype=torch.long).unsqueeze(0)
            if 'aux_features_contrastive' not in x:
                x['aux_features_contrastive'] = absolute
            else:
                raise

        return x, y


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
    if core_leftover > 0:
        offset = torch.randint(minleader, minleader + core_leftover, (1,)).item()
    else:
        offset = minleader

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
    hd_matrix = torch.zeros((len(msa), len(msa)))

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
    hd_matrix = torch.zeros((len(msa_indices), len(sampled_msa)))

    for idx_1, idx_msa in enumerate(msa_indices):
        seq_1 = msa[idx_msa]
        for idx_2, seq_2 in enumerate(sampled_msa):
            hd_matrix[idx_1, idx_2] = _hamming_distance(seq_1.seq, seq_2.seq)

    # average over already sampled sequences
    avg_hd = torch.mean(hd_matrix, dim=1)
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
        avg_hd = torch.mean(hd_matrix_reduced, dim=1)
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
