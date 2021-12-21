from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import torch
from Bio.Align import MultipleSeqAlignment
import math
import random

from selbstaufsicht.utils import lehmer_encode

# TODO task scheduling a la http://bmvc2018.org/contents/papers/0345.pdf


class MSATokenize():
    def __init__(self, mapping: Dict[str, int]) -> None:
        """
        Initializes MSA tokenizing transform.

        Args:
            mapping (Dict[str, int]): Mapping from the lettered input alphabet to its numerical token representation.
        """

        self.mapping = mapping

    def __call__(self, x: Dict[str, MultipleSeqAlignment], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs tokenization, i.e., replaces each character token in the MSA by its numerical token representation, according to the given mapping.

        Args:
            x (Dict[str, MultipleSeqAlignment]): Lettered MSA.
            y (Dict[str, torch.Tensor]): Upstream task labels.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: x: Tokenized MSA [E, L]; y: Upstream task labels.
        """

        x['msa'] = torch.tensor([[self.mapping[letter] for letter in sequence.upper()] for sequence in x['msa']], dtype=torch.long)
        if 'contrastive' in x:
            x['contrastive'] = torch.tensor([[self.mapping[letter] for letter in sequence.upper()] for sequence in x['contrastive']], dtype=torch.long)

        if 'START_TOKEN' in self.mapping:
            prefix = torch.full((x['msa'].size(0), 1), self.mapping['START_TOKEN'], dtype=torch.int)
            x['msa'] = torch.cat([prefix, x['msa']], dim=-1)
            if 'contrastive' in x:
                prefix = torch.full((x['contrastive'].size(0), 1), self.mapping['START_TOKEN'], dtype=torch.int)
                x['contrastive'] = torch.cat([prefix, x['contrastive']], dim=-1)

        return x, y


class MSACropping():
    def __init__(self, length: int, contrastive: bool = False, mode: str = 'random-dependent') -> None:
        """
        Initializes MSA cropping transform.

        Args:
            length (int): Maximum uncropped sequence length.
            contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.
            mode (str, optional): Cropping mode. Currently implemented: random-dependent, random-independent, fixed. Defaults to 'random-dependent'.
        """

        self.length = length
        self.contrastive = contrastive
        self.cropping_fn = _get_msa_cropping_fn(mode)

    def __call__(self, x: Dict[str, MultipleSeqAlignment], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, MultipleSeqAlignment], Dict[str, torch.Tensor]]:
        """
        Crops each sequence of the given lettered MSA randomly to the predefined length.

        Args:
            x (Dict[str, MultipleSeqAlignment]): Lettered MSA.
            y (Dict[str, torch.Tensor]): Upstream task labels.

        Returns:
            Tuple[Dict[str, MultipleSeqAlignment], Dict[str, torch.Tensor]]: x: Cropped, lettered MSA; y: Upstream task labels.
        """

        msa = x['msa'][:, :]
        if x['msa'].get_alignment_length() > self.length:
            x['msa'] = self.cropping_fn(x['msa'], self.length, False)

        if self.contrastive:
            contrastive_msa = x.get('contrastive', msa)
            if contrastive_msa.get_alignment_length() > self.length:
                contrastive_msa = self.cropping_fn(contrastive_msa, self.length, True)
            x['contrastive'] = contrastive_msa
        return x, y


class RandomMSAMasking():
    # TODO other tokens instead of mask token
    def __init__(self, p: float, mode: str, mask_token: int, contrastive: bool = False, start_token: bool = True) -> None:
        """
        Initializes random MSA masking transform.

        Args:
            p (float): Masking probability.
            mode (str): Masking mode. Currently implemented: block-wise, column-wise, token-wise.
            mask_token (int): Masking token.
            contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.
            start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.
        """

        self.p = p
        self.mask_token = mask_token
        self.masking_fn = _get_masking_fn(mode, start_token)
        self.contrastive = contrastive

    def __call__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs random masking on the given MSA, according to the predefined masking probability and mode.

        Args:
            x (Dict[str, torch.Tensor]): Tokenized MSA [E, L].
            y (Dict[str, torch.Tensor]): Upstream task labels.

        Raises:
            ValueError: Unexpected input data type.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: x: Masked, tokenized MSA [E, L]; Masking mask [E, L]; y: Inpainting label
                (flattened tensor of masked tokens) [~p*E*L].
        """

        masked_msa, mask, target = self.masking_fn(x['msa'], self.p, self.mask_token)
        x['msa'] = masked_msa
        x['mask'] = mask
        y['inpainting'] = target
        if self.contrastive:
            x['contrastive'], _, _ = self.masking_fn(x['contrastive'], self.p, self.mask_token)
        return x, y


class RandomMSAShuffling():
    def __init__(self,
                 permutations: torch.Tensor = None,
                 minleader: int = 1,
                 mintrailer: int = 0,
                 delimiter_token: int = None,
                 num_partitions: int = None,
                 num_classes: int = None,
                 contrastive: bool = False):
        """
        Initializes random MSA shuffling.

        Args:
            permutations (torch.Tensor, optional): Explicitly specified permutations [NClasses, NPartitions].
                Is inferred from \"num_partitions\" and \"num_classes\" otherwise. Defaults to None.
            minleader (int, optional): Minimum number of unshuffled tokens at the start of each sequence. Defaults to 1.
            mintrailer (int, optional): Minimum number of unshuffled tokens at the end of each sequence. Defaults to 0.
            delimiter_token (int, optional): Special token that is used to separate shuffled partitions from each other. Defaults to None.
            num_partitions (int, optional): Number of shuffled partitions per sequence. Needs to be specified, if \"permutations\" is unspecified. Defaults to None.
            num_classes (int, optional): Number of allowed permutations. Needs to be specified, if \"permutations\" is unspecified. Defaults to None.
            contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

        Raises:
            ValueError: Either \"permutations\" or \"num_partitions\" and \"num_classes\" need to be specified.
        """

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

    def __call__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs random shuffling on the given MSA, according to the predefined allowed permutations.

        Args:
            x (Dict[str, torch.Tensor]): Tokenized MSA [E, L].
            y (Dict[str, torch.Tensor]): Upstream task labels.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: x: Shuffled, tokenized MSA [E, L]; y: Jigsaw label (permutation index per sequence) [E].
        """

        num_seq = x['msa'].size(0)
        if 'jigsaw' in y:
            # TODO: ugly, only works for fixed subsampling mode
            label = y['jigsaw'][:num_seq]
        else:
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


class MSASubsampling():
    def __init__(self, num_sequences: int, contrastive: bool = False, mode: str = 'uniform') -> None:
        """
        Initializes MSA subsampling.

        Args:
            num_sequences (int): Number of subsampled sequences.
            contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.
            mode (str, optional): Subsampling mode. Currently implemented: uniform, diversity, fixed. Defaults to 'uniform'.
        """

        self.contrastive = contrastive
        self.sampling_fn = _get_msa_subsampling_fn(mode)
        self.nseqs = num_sequences

    def __call__(self, x: Dict[str, MultipleSeqAlignment], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, MultipleSeqAlignment], Dict[str, torch.Tensor]]:
        """
        Subsamples the predefined number of sequences from the given lettered MSA, according to the predefined subsampling mode.

        Args:
            x (Dict[str, MultipleSeqAlignment]): Lettered MSA.
            y (Dict[str, torch.Tensor]): Upstream task labels.

        Returns:
            Tuple[Dict[str, MultipleSeqAlignment], Dict[str, torch.Tensor]]: x: Subsampled lettered MSA; y: Upstream task labels.
        """

        msa = x['msa'][:, :]
        x['msa'] = self.sampling_fn(msa, self.nseqs, False)
        if self.contrastive:
            x['contrastive'] = self.sampling_fn(msa, self.nseqs, True)
        return x, y


# TODO this is sort of a hacky way to fix the incorrect way of PE without having to touch everything
class ExplicitPositionalEncoding():
    def __init__(self, max_seqlen=5000):
        """
        Initializes explicite positional encoding.

        Args:
            max_seqlen (int, optional): longest sequence length allowed. Defaults to 5000.
        """
        self.max_seqlen = max_seqlen

    def __call__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs explicit positional encoding to create absolute and relative positional auxiliary features.

        Args:
            x (Dict[str, torch.Tensor]): Tokenized MSA [E, L].
            y (Dict[str, torch.Tensor]): Upstream task labels.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            x: Tokenized MSA [E, L]; absolute and relative positional auxiliary features [1, L];
            y: Upstream task labels.
        """

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


class DistanceFromChain():

    def __call__(self, x: Dict, y: Dict) -> Tuple[Dict, Dict]:
        """
        Takes a biopython structure containing a single chain and returns a distance map.
        Args:
            structure (Bio.PDB.Structure): Molecular structure to generate distance map from.
        Returns:
            torch.Tensor [L', L'] residue distance map
        """
        structure = y['structure']
        # TODO missing residue check?
        assert len(structure) == 1
        assert len(structure[0]) == 1

        def mindist(r1, r2):
            distances = torch.tensor([[a1 - a2 for a2 in r2] for a1 in r1])
            return torch.min(distances)

        chain = structure[0].get_list()[0]
        distances = torch.zeros((len(chain), len(chain)))
        for i, res1 in enumerate(chain.get_list()):
            for j in range(i + 1, len(chain)):
                res2 = chain.get_list()[j]
                distances[i, j] = mindist(res1, res2)

        distances = distances + distances.t()
        y['distances'] = distances
        return x, y


class ContactFromDistance():
    def __init__(self, threshold: float = 4.):
        """
        Thresholds used in CoCoNet paper were 4. and 10. Angstrom
        """
        self.threshold = threshold

    def __call__(self, x: Dict, y: Dict) -> Tuple[Dict, Dict]:
        y['distances'] = y['distances'] < self.threshold
        return x, y


# TODO maybe remove possible shortcut of e.g.
# >AAA|BB
# >BB|AAA
# should the leader and trailer be adapted such, that all partitions are of the same size?

def _jigsaw(msa: torch.Tensor, permutations: torch.Tensor, delimiter_token: int = None, minleader: int = 1, mintrailer: int = 0) -> torch.Tensor:
    """
    Shuffles the given MSA according to the given permutations.

    Args:
        msa (torch.Tensor): Tokenized MSA to be shuffled [E, L].
        permutations (torch.Tensor): Permutations to be applied sequence-wise to the MSA [E, NPartitions].
        delimiter_token (int, optional): Special token that is used to separate shuffled partitions from each other. Defaults to None.
        minleader (int, optional): Minimum number of unshuffled tokens at the start of each sequence. Defaults to 1.
        mintrailer (int, optional): Minimum number of unshuffled tokens at the end of each sequence. Defaults to 0.

    Returns:
        torch.Tensor: Shuffled, tokenized MSA [E, L].
    """

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
def _block_mask_msa(msa: torch.Tensor, p: float, mask_token: int, start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out a contiguous block of columns in the given MSA, whose size is determined by the given probability/ratio.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        p (float): Masking probability/ratio.
        mask_token (int): Special token that is used for masking.
        start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L]
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


def _column_mask_msa_indexed(msa: torch.Tensor, col_indices: torch.Tensor, mask_token: int, start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out a given set of columns in the given MSA.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        col_indices (torch.Tensor): Indices of columns that are to be masked.
        mask_token (int): Special token that is used for masking.
        start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L]
    """

    mask = torch.zeros_like(msa, dtype=torch.bool)
    mask[:, col_indices + int(start_token)] = True

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _column_mask_msa(msa: torch.Tensor, p: float, mask_token: int, start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out a random set of columns in the given MSA, whose size is determined by the given probability/ratio.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        p (float): Masking probability/ratio.
        mask_token (int): Special token that is used for masking.
        start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L]
    """

    col_num = msa.size(-1) - int(start_token)
    col_indices = torch.arange(col_num, dtype=torch.long)
    col_mask = torch.full((col_num,), p)
    col_mask = torch.bernoulli(col_mask).to(torch.bool)
    masked_col_indices = col_indices[col_mask]
    return _column_mask_msa_indexed(msa, masked_col_indices, mask_token, start_token=start_token)


# TODO should seq start token be excluded from masking?
def _token_mask_msa(msa: torch.Tensor, p: float, mask_token: int, start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out random tokens uniformly sampled from the given MSA, according to the given probability/ratio.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        p (float): Masking probability/ratio.
        mask_token (int): Special token that is used for masking.
        start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L]
    """

    mask = torch.full(msa.size(), p)
    mask[:, :int(start_token)] = 0.
    mask = torch.bernoulli(mask).to(torch.bool)

    masked = msa[mask]
    msa[mask] = mask_token
    return msa, mask, masked


def _get_masking_fn(mode: str, start_token: bool) -> Callable[[torch.Tensor, float, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Returns the masking function that corresponds to the given masking mode.

    Args:
        mode (str): Masking mode. Currently implemented: block-wise, column-wise, token-wise.
        start_token (bool): Whether a start token is used, which is then precluded from masking.

    Raises:
        ValueError: Unknown masking mode.

    Returns:
        Callable[[torch.Tensor, float, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Masking function
        (tokenized MSA [E, L]; masking probability/ratio; masking token -> masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L])
    """

    if mode == 'token':
        return partial(_token_mask_msa, start_token=start_token)
    elif mode == 'column':
        return partial(_column_mask_msa, start_token=start_token)
    elif mode == 'block':
        return partial(_block_mask_msa, start_token=start_token)
    raise ValueError('unknown token masking mode', mode)


def _subsample_uniform(msa: MultipleSeqAlignment, nseqs: int, contrastive: bool = False) -> MultipleSeqAlignment:
    """
    Subsamples sequences uniformly sampled from the given MSA.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        nseqs (int): Number of sequences to be subsampled.
        contrastive (bool): Whether contrastive learning is active. Defaults to False.

    Returns:
        MultipleSeqAlignment: Subsampled, lettered MSA.
    """

    max_nseqs = len(msa)
    if max_nseqs > nseqs:
        indices = torch.randperm(max_nseqs)[:nseqs]
        msa = MultipleSeqAlignment([msa[i.item()] for i in indices])
    return msa


def _hamming_distance(seq_1: str, seq_2: str) -> int:
    """
    Computes the hamming distance, i.e., the number of index-wise different characters, for the two given sequences.

    Args:
        seq_1 (str): First sequence.
        seq_2 (str): Second sequence.

    Returns:
        int: Hamming distance.
    """

    assert len(seq_1) == len(seq_2), "Both sequences are required to have the same length!"
    return sum(n_1 != n_2 for n_1, n_2 in zip(seq_1, seq_2))


def _hamming_distance_matrix(msa: MultipleSeqAlignment) -> torch.Tensor:
    """
    Computes hamming distances between all pairs of different sequences from the given MSA.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.

    Returns:
        torch.Tensor: Symmetric, zero-diagonal matrix with sequence-to-sequence hamming distances [E, E].
    """

    hd_matrix = torch.zeros((len(msa), len(msa)))

    # computes upper triangular part, without diagonal
    for idx_1, seq_1 in enumerate(msa):
        for idx_2, seq_2 in enumerate(msa):
            if idx_2 <= idx_1:
                continue
            hd_matrix[idx_1, idx_2] = _hamming_distance(seq_1.seq, seq_2.seq)

    # make matrix symmetric for easier handling
    return hd_matrix + hd_matrix.T


def _maximize_diversity_naive(msa: MultipleSeqAlignment, msa_indices: List[int], nseqs: int, sampled_msa: MultipleSeqAlignment) -> MultipleSeqAlignment:
    """
    Subsamples sequences from the given MSA according to the greedy diviserty maximization scheme.
    This function uses the naive strategy, where hamming distances between sequences are computed on-the-fly when needed, potentially repeatedly.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        msa_indices (List[int]): Indices of remaining (not already subsampled) sequences.
        nseqs (int): Number of sequences still to be subsampled.
        sampled_msa (MultipleSeqAlignment): Already subsampled sequences.

    Returns:
        MultipleSeqAlignment: Subsampled, lettered MSA.
    """

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


def _maximize_diversity_cached(msa: MultipleSeqAlignment,
                               msa_indices: List[int],
                               nseqs: int,
                               sampled_msa: MultipleSeqAlignment,
                               sampled_msa_indices: List[int],
                               hd_matrix: torch.Tensor) -> MultipleSeqAlignment:
    """
    Subsamples sequences from the given MSA according to the greedy diviserty maximization scheme.
    This function uses the cached strategy, where hamming distances between all non-reflexive sequences-to-sequence pairs are computed beforehand and cached.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        msa_indices (List[int]): Indices of remaining (not already subsampled) sequences.
        nseqs (int): Number of sequences still to be subsampled.
        sampled_msa (MultipleSeqAlignment): Already subsampled sequences.
        sampled_msa_indices (List[int]): Indices of already subsampled sequences.
        hd_matrix (torch.Tensor): Symmetric matrix with sequence-to-sequence hamming distances [E, E].

    Returns:
        MultipleSeqAlignment: Subsampled, lettered MSA.
    """

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


def _subsample_diversity_maximizing(msa: MultipleSeqAlignment, nseqs: int, contrastive: bool = False) -> MultipleSeqAlignment:
    """
    Subsamples sequences from the given MSA according to the greedy diviserty maximization scheme.
    Depending on the number of sequences in the given MSA and the number of sequences to be subsampled, it chooses the more efficient computation strategy automatically.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        nseqs (int): Number of sequences to be subsampled.
        contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

    Returns:
        MultipleSeqAlignment: Subsampled, lettered MSA.
    """

    # since the function is deterministic and contrastive input should be different from the regular input, it is sampled randomly
    if contrastive:
        return _subsample_uniform(msa, nseqs)

    # depending on the total number of sequences and the number of sequences to be subsampled, choose computation strategy:
    # either compute and cache all hamming distanced between distinct sequences beforehand or use the naive implementation with potentially repeating comparisons
    # TODO: Combine naive and chached strategies to optimal strategy: Compute hamming distances on-the-fly when needed, but cache for later re-use

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


def _subsample_fixed(msa: MultipleSeqAlignment, nseqs: int, contrastive: bool = False) -> MultipleSeqAlignment:
    """
    Subsamples the first n sequences from the MSA.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        nseqs (int): Number of sequences to be subsampled.
        contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

    Returns:
        MultipleSeqAlignment: Subsampled, lettered MSA.
    """

    if contrastive:
        return msa[nseqs:2 * nseqs]
    else:
        return msa[:nseqs]


def _get_msa_subsampling_fn(mode: str) -> Callable[[MultipleSeqAlignment, int, Optional[bool]], MultipleSeqAlignment]:
    """
    Returns the subsampling function that corresponds to the given subsampling mode.

    Args:
        mode (str): Subsampling mode. Currently implemented: uniform, diversity, fixed.

    Raises:
        ValueError: Unknown subsampling mode.

    Returns:
        Callable[[MultipleSeqAlignment, int, Optional[bool]], MultipleSeqAlignment]: Subsampling function
        (lettered MSA; number of sequences to be subsampled; whether contrastive lerning is active -> subsampled, lettered MSA)
    """

    if mode == 'uniform':
        return _subsample_uniform
    if mode == 'diversity':
        return _subsample_diversity_maximizing
    if mode == 'fixed':
        return _subsample_fixed
    raise ValueError('unkown msa sampling mode', mode)


def _crop_random_dependent(msa: MultipleSeqAlignment, length: int, contrastive: bool = False) -> MultipleSeqAlignment:
    """
    Crops each sequence of the given lettered MSA randomly to the predefined length.
    Cropping start is the same for all sequences.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        length (int): Maximum uncropped sequence length.
        contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

    Returns:
        MultipleSeqAlignment: Cropped, lettered MSA.
    """

    start = torch.randint(msa.get_alignment_length() - length, (1,)).item()
    return msa[:, start: start + length]


def _crop_random_independent(msa: MultipleSeqAlignment, length: int, contrastive: bool = False) -> MultipleSeqAlignment:
    """
    Crops each sequence of the given lettered MSA randomly to the predefined length.
    Cropping start is randomly sampled for all sequences.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        length (int): Maximum uncropped sequence length.
        contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

    Returns:
        MultipleSeqAlignment: Cropped, lettered MSA.
    """

    starts = torch.randint(msa.get_alignment_length() - length, (len(msa),))
    cropped_msa = MultipleSeqAlignment([])
    for idx in range(len(msa)):
        start = starts[idx].item()
        cropped_msa.append(msa[idx, start: start + length])
    return cropped_msa


def _crop_fixed(msa: MultipleSeqAlignment, length: int, contrastive: bool = False) -> MultipleSeqAlignment:
    """
    Crops each sequence of the given lettered MSA in a left-aligned way to the predefined length.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        length (int): Maximum uncropped sequence length.
        contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

    Returns:
        MultipleSeqAlignment: Cropped, lettered MSA.
    """

    return msa[:, :length]


def _get_msa_cropping_fn(mode: str) -> Callable[[MultipleSeqAlignment, int, Optional[bool]], MultipleSeqAlignment]:
    """
    Returns the cropping function that corresponds to the given cropping mode.

    Args:
        mode (str): Cropping mode. Currently implemented: random-dependent, random-independent, fixed.

    Raises:
        ValueError: Unknown cropping mode.

    Returns:
        Callable[[MultipleSeqAlignment, int, Optional[bool]], MultipleSeqAlignment]: Cropping function
        (lettered MSA; maximum uncropped sequence length; whether contrastive lerning is active -> cropped, lettered MSA)
    """

    if mode == 'random-dependent':
        return _crop_random_dependent
    if mode == 'random-independent':
        return _crop_random_independent
    if mode == 'fixed':
        return _crop_fixed
    raise ValueError('unkown msa cropping mode', mode)
