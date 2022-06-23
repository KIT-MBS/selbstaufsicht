from functools import partial
from typing import Callable, Dict, List, Optional, Union, Tuple
import torch
from torch.distributions import Categorical, Distribution
import numpy as np
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
    def __init__(self, p: float, p_static: float, p_nonstatic: float, p_unchanged: float, mode: str, static_mask_token: int,
                 nonstatic_mask_tokens: List[int], contrastive: bool = False, start_token: bool = True) -> None:
        """
        Initializes random MSA masking transform.

        Args:
            p (float): Masking probability.
            p_static (float): Conditional probability for static inpainting, if masked.
            p_nonstatic (float): Conditional probability for nonstatic inpainting, if masked.
            p_unchanged (float): Conditional probability for no change, if masked.
            mode (str): Masking mode. Currently implemented: block-wise, column-wise, token-wise.
            static_mask_token (int): Token that is used to replace masked tokens static masking.
            nonstatic_mask_tokens (List[int]): Tokens that are used to replace masked tokens randomly in nonstatic masking.
            contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.
            start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.
        """

        self.p = p
        assert p_static + p_nonstatic + p_unchanged == 1
        self.masking_type_distribution = Categorical(torch.tensor([p_unchanged, p_static, p_nonstatic]))
        self.static_mask_token = static_mask_token
        self.nonstatic_mask_tokens = nonstatic_mask_tokens
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

        masked_msa, mask, target = self.masking_fn(x['msa'], self.p, self.masking_type_distribution,
                                                   self.static_mask_token, self.nonstatic_mask_tokens)
        x['msa'] = masked_msa
        x['mask'] = mask
        y['inpainting'] = target
        if self.contrastive:
            x['contrastive'], _, _ = self.masking_fn(x['contrastive'], self.p, self.masking_type_distribution,
                                                     self.static_mask_token, self.nonstatic_mask_tokens)
        return x, y


class RandomMSAShuffling():
    def __init__(self,
                 permutations: torch.Tensor = None,
                 minleader: int = 1,
                 mintrailer: int = 0,
                 delimiter_token: int = None,
                 num_partitions: int = None,
                 num_classes: int = None,
                 euclid_emb: torch.Tensor = None,
                 contrastive: bool = False,
                 frozen: bool = False):
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
            euclid_emb (torch.Tensor, optional): Euclidean embedding of the discrete permutation metric. Defaults to None.
            contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

        Raises:
            ValueError: Either \"permutations\" or \"num_partitions\" and \"num_classes\" need to be specified.
        """

        self.frozen = frozen
        if permutations is None and (num_partitions is None or num_classes is None):
            raise ValueError("Permutations have to be given explicitely or parameters to generate them.")

        if permutations is None:
            self.perm_indices = list(range(1, math.factorial(num_partitions)))
            random.shuffle(self.perm_indices)
            # NOTE always include 'no transformation'
            self.perm_indices.insert(0, 0)
            self.permutations = torch.stack([lehmer_encode(i, num_partitions) for i in self.perm_indices[:num_classes]]).unsqueeze(0)
            self.perm_indices = torch.tensor(self.perm_indices)
        else:
            # NOTE attribute permutations is expected to have shape [num_classes, num_partitions]
            # NOTE add singleton dim for later expansion to num_seq dim
            self.perm_indices = None
            self.permutations = permutations.unsqueeze(0)
        self.num_partitions = max(self.permutations[0, 0])
        self.num_classes = len(self.permutations[0])
        self.minleader = minleader
        self.mintrailer = mintrailer
        self.delimiter_token = delimiter_token
        self.euclid_emb = euclid_emb
        self.euclid_emb_device_flag = False
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
            # NOTE: ugly, only works for fixed subsampling mode
            if self.frozen:
                perm_sampling = y['jigsaw'][0]
            else:
                perm_sampling = y['jigsaw'][:num_seq]
        else:
            if self.frozen:
                perm_sampling = torch.randint(0, self.num_classes, (1,))*torch.ones((num_seq,))
                perm_sampling = perm_sampling.type(torch.LongTensor)
            else:
                perm_sampling = torch.randint(0, self.num_classes, (num_seq,))
        shuffled_msa = _jigsaw(x['msa'],
                               self.permutations.expand(num_seq, -1, -1)[range(num_seq),
                               perm_sampling], delimiter_token=self.delimiter_token,
                               minleader=self.minleader,
                               mintrailer=self.mintrailer)
        x['msa'] = shuffled_msa
        if self.euclid_emb is None:

            if self.frozen:
                y['jigsaw']=perm_sampling[0]
                y['jigsaw']=y['jigsaw'].reshape(1)
            else:
                y['jigsaw'] = perm_sampling
        else:
            if self.perm_indices is None:
                # TODO: Implement inverse lehmer code to solve?
                raise TypeError("Fixed permutations and euclidean embedding are not compatible!")
            # necessary for pytorch lightning to push the tensor onto the correct cuda device
            if not self.euclid_emb_device_flag:
                self.euclid_emb = self.euclid_emb.type_as(x['msa']).float()
                self.euclid_emb_device_flag = True
            if self.frozen:
                y['jigsaw'] = self.euclid_emb[self.perm_indices[perm_sampling], :][0]
            else:
                y['jigsaw'] = self.euclid_emb[self.perm_indices[perm_sampling], :]

        if self.contrastive:
            contrastive_perm_sampling = torch.randint(0, self.num_classes, (num_seq,))
            x['contrastive'] = _jigsaw(x['contrastive'],
                                       self.permutations.expand(num_seq, -1, -1)[range(num_seq),
                                       contrastive_perm_sampling],
                                       delimiter_token=self.delimiter_token,
                                       minleader=self.minleader,
                                       mintrailer=self.mintrailer)
        return x, y


class MSAboot():

    def __init__(self, ratio: float = 0.5, per_token: bool = False, boot_same: bool = False, seq_dist: bool = False) -> None:

        self.ratio = ratio
        self.per_token = per_token
        self.boot_same = boot_same
        self.seq_dist = seq_dist

    def __call__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        num_seq = x['msa'].shape[0]
        num_col = x['msa'].shape[1]

        ind_rand = torch.ones((int(np.floor(self.ratio*num_seq)), num_seq)).multinomial(num_samples=num_col, replacement=True)

        msa_boot = _jigsaw_boot(x['msa'], self.ratio, ind_rand)

        msa_boot_expanded = msa_boot.unsqueeze(1).expand(num_seq, num_seq, num_col)
        msa_repeated = x['msa'].repeat(num_seq, 1, 1)

        msa_bool = torch.eq(msa_boot_expanded, msa_repeated)

        if not self.per_token:
            y_boot = torch.any(torch.all(msa_bool, keepdim=True, axis=2), axis=1).reshape(num_seq,)

            y['jigsaw_boot'] = y_boot.type(torch.LongTensor)
        else:
            if not self.boot_same:
                # y_ind=torch.argmax(torch.sum(msa_bool,keepdim=True,axis=2),axis=1)

                y_boot = torch.eq(msa_boot, x['msa'][torch.argmax(torch.sum(msa_bool, keepdim=True, axis=2), axis=1)].reshape(num_seq, num_col))

            else:
                y_boot = torch.eq(msa_boot, x['msa'])

            if self.seq_dist:

                y_boot = y_boot.sum(axis=1).div(y_boot.shape[1])
                y_boot = y_boot.type(torch.FloatTensor)

                y['jigsaw_boot'] = y_boot.reshape(num_seq,)
            else:

                y_boot = y_boot.type(torch.LongTensor)

                y['jigsaw_boot'] = y_boot.reshape(num_col*num_seq,)

        x['msa'] = msa_boot

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
        self.mode = mode
        self.sampling_fn = _get_msa_subsampling_fn(mode)
        self.nseqs = num_sequences

    def __call__(self, x: Dict[str, Union[MultipleSeqAlignment, torch.Tensor]], y: Dict[str, torch.Tensor]) -> Tuple[Dict[str, MultipleSeqAlignment], Dict[str, torch.Tensor]]:
        """
        Subsamples the predefined number of sequences from the given lettered MSA, according to the predefined subsampling mode.

        Args:
            x (Dict[str, Union[MultipleSeqAlignment, torch.Tensor]]): Lettered MSA; Subsampling indices [num_seq] (if mode=diversity).
            y (Dict[str, torch.Tensor]): Upstream task labels.

        Returns:
            Tuple[Dict[str, MultipleSeqAlignment], Dict[str, torch.Tensor]]: x: Subsampled lettered MSA; y: Upstream task labels.
        """

        msa = x['msa'][:, :]
        if self.mode == 'diversity':
            if 'indices' not in x:
                raise KeyError('No indices provided for diversity-maximizing subsampling!')
            x['msa'] = self.sampling_fn(msa, self.nseqs, x['indices'], False)
            del x['indices']
        else:
            x['msa'] = self.sampling_fn(msa, self.nseqs, False)
        if self.contrastive:
            # diversity maximization should not be used in combination with contrastive
            assert self.mode != 'diversity'
            x['contrastive'] = self.sampling_fn(msa, self.nseqs, True)
        return x, y


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
    def __init__(self, chunk_size: int = 600, device=None):
        """
        Initializes distance-from-chain computation.

        Args:
            chunk_size (int, optional): Chunk size in residuum dimension. Defaults to 600.
        """

        self.device = device
        self.chunk_size = chunk_size

    def __call__(self, x: Dict, y: Dict) -> Tuple[Dict, Dict]:
        """
        Takes a biopython structure containing a single chain and returns a distance map.

        Args:
            structure (Bio.PDB.Structure): Molecular structure to generate distance map from.

        Returns:
            torch.Tensor [L', L'] residue distance map
        """
        structure = y['structure']
        assert len(structure) == 1
        assert len(structure[0]) == 1

        if self.device is None:
            if torch.cuda.is_available():
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:
                    device = 'cuda'
                else:
                    # only as many data loaders as GPUs are allowed
                    assert worker_info.num_workers <= torch.cuda.device_count()
                    device = 'cuda:%d' % worker_info.id
            else:
                device = 'cpu'
        else:
            device = self.device

        chain = structure[0].get_list()[0]
        atom_coords = [[a.get_coord() for a in r] for r in chain]
        nums_atoms = [len(atom_coords_r) for atom_coords_r in atom_coords]
        num_atoms_max = max(nums_atoms)
        num_residues = len(atom_coords)
        num_chunks = int(math.ceil(num_residues / self.chunk_size))

        distances = torch.zeros((num_residues, num_residues), device=device)
        atom_coords_t = torch.full((num_residues, num_atoms_max, 3), torch.inf, device=device)  # [R, A, 3]
        for idx in range(num_residues):
            atom_coords_rt = torch.tensor(np.array(atom_coords[idx]), device=device)
            atom_coords_t[idx, :nums_atoms[idx], :] = atom_coords_rt

        for idx in range(num_chunks):
            residues_slice = slice(idx * self.chunk_size, (idx + 1) * self.chunk_size)

            atom_coords_t1 = atom_coords_t[residues_slice, :, :].view(-1, 1, num_atoms_max, 1, 3).expand(-1, num_residues, num_atoms_max, num_atoms_max, 3)  # [RC, R, A, A, 3]
            atom_coords_t2 = atom_coords_t.view(1, num_residues, 1, num_atoms_max, 3).expand(atom_coords_t1.shape[0], num_residues, num_atoms_max, num_atoms_max, 3)  # [RC, R, A, A, 3]

            distances_chunk = torch.linalg.vector_norm(atom_coords_t1 - atom_coords_t2, dim=-1)  # [RC, R, A, A]
            distances_chunk = torch.nan_to_num(distances_chunk, nan=torch.inf, posinf=torch.inf)
            distances_chunk = torch.amin(distances_chunk, dim=(-1, -2))  # [RC, R]
            distances[residues_slice, :] = distances_chunk

        del y['structure']
        y['distances'] = distances
        return x, y


class ContactFromDistance():
    def __init__(self, threshold: float = 4.):
        """
        Thresholds used in CoCoNet paper were 4. and 10. Angstrom
        """
        self.threshold = threshold

    def __call__(self, x: Dict, y: Dict) -> Tuple[Dict, Dict]:
        contacts = torch.zeros_like(y['distances'], dtype=torch.long)
        contacts[y['distances'] < self.threshold] = 1.
        contacts[y['distances'] == torch.inf] = -1.

        y['contact'] = contacts
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


def _jigsaw_boot(msa: torch.Tensor, ratio: float, ind_rand: torch.Tensor) -> torch.Tensor:
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

    msa_copy = torch.clone(msa)

    num_seq = msa.shape[0]
    num_seq_r = int(np.floor(num_seq*ratio))
    num_col = msa.shape[1]

    ind_col = torch.arange(0, num_col)
    ind_rand_r = torch.randperm(num_seq)[range(num_seq_r)]

    msa_copy[ind_rand_r, ] = msa[ind_rand, ind_col]

    return msa_copy


def _get_replace_mask(mask: torch.Tensor, masking_type_sampling: torch.Tensor, static_mask_token: int,
                      nonstatic_mask_tokens: List[int]) -> torch.Tensor:
    """
    Creates replace mask, which randomly assigns a mask token to each token to be masked.

    Args:
        mask (torch.Tensor): Boolean inpainting mask.
        masking_type_sampling (torch.Tensor): Sampling from categorical distribution of conditional probabilities for each masking type.
        static_mask_token (int): Token that is used to replace masked tokens static masking.
        nonstatic_mask_tokens (List[int]): Tokens that are used to replace masked tokens randomly in nonstatic masking.

    Returns:
        torch.Tensor: Replace mask, which randomly assigns a mask token to each token to be masked.
    """

    nonstatic_mask_tokens_t = torch.tensor(nonstatic_mask_tokens)

    assert set(torch.unique(masking_type_sampling).tolist()).issubset({0, 1, 2})
    replace_mask = masking_type_sampling  # [0: unchanged, 1: static, 2: nonstatic]
    replace_mask *= mask
    # exploiting call-by-reference
    mask[replace_mask == 0] = False
    replace_mask -= 2  # [-2: unchanged, -1: static, 0: nonstatic]

    # inserting nonstatic tokens
    num_nonstatic = replace_mask[replace_mask == 0].numel()
    nonstatic_sampling = torch.randint(0, len(nonstatic_mask_tokens), (num_nonstatic,))
    replace_mask[replace_mask == 0] = nonstatic_sampling  # [-2: unchanged, -1: static, 0: nonstatic_1, ..., n-1: nonstatic_n]
    # leveraging that tokens are non-negative
    replace_mask[replace_mask >= 0] = nonstatic_mask_tokens_t[nonstatic_sampling]  # [-2: unchanged, -1: static] + nonstatic_tokens

    # inserting static token
    replace_mask[replace_mask == -1] = static_mask_token  # [-2: unchanged] + [static_token] + nonstatic_tokens

    # discarding unchanged positions
    replace_mask = replace_mask[replace_mask != -2]  # [static_token] + nonstatic_tokens

    return replace_mask


def _block_mask_msa(msa: torch.Tensor, p: float, masking_type_distribution: Distribution, static_mask_token: int, nonstatic_mask_tokens: List[int],
                    start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out a contiguous block of columns in the given MSA, whose size is determined by the given probability/ratio.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        p (float): Masking probability/ratio.
        masking_type_distribution (Distribution): Categorical distribution of conditional probabilities for each masking type.
        static_mask_token (int): Token that is used to replace masked tokens static masking.
        nonstatic_mask_tokens (List[int]): Tokens that are used to replace masked tokens randomly in nonstatic masking.
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

    masking_type_sampling = masking_type_distribution.sample(mask.size())
    replace_mask = _get_replace_mask(mask, masking_type_sampling, static_mask_token, nonstatic_mask_tokens)

    masked = msa[mask]
    msa[mask] = replace_mask
    return msa, mask, masked


def _column_mask_msa_indexed(msa: torch.Tensor, col_indices: torch.Tensor, masking_type_distribution: Distribution, static_mask_token: int,
                             nonstatic_mask_tokens: List[int], start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out a given set of columns in the given MSA.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        col_indices (torch.Tensor): Indices of columns that are to be masked.
        masking_type_distribution (Distribution): Categorical distribution of conditional probabilities for each masking type.
        static_mask_token (int): Token that is used to replace masked tokens static masking.
        nonstatic_mask_tokens (List[int]): Tokens that are used to replace masked tokens randomly in nonstatic masking.
        start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L]
    """

    mask = torch.zeros_like(msa, dtype=torch.bool)
    mask[:, col_indices + int(start_token)] = True

    masking_type_sampling = masking_type_distribution.sample(mask.size())
    replace_mask = _get_replace_mask(mask, masking_type_sampling, static_mask_token, nonstatic_mask_tokens)

    masked = msa[mask]
    msa[mask] = replace_mask
    return msa, mask, masked


def _column_mask_msa(msa: torch.Tensor, p: float, masking_type_distribution: Distribution, static_mask_token: int, nonstatic_mask_tokens: List[int],
                     start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out a random set of columns in the given MSA, whose size is determined by the given probability/ratio.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        p (float): Masking probability/ratio.
        masking_type_distribution (Distribution): Categorical distribution of conditional probabilities for each masking type.
        static_mask_token (int): Token that is used to replace masked tokens static masking.
        nonstatic_mask_tokens (List[int]): Tokens that are used to replace masked tokens randomly in nonstatic masking.
        start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L]
    """

    col_num = msa.size(-1) - int(start_token)
    col_indices = torch.arange(col_num, dtype=torch.long)
    col_mask = torch.full((col_num,), p)
    col_mask = torch.bernoulli(col_mask).to(torch.bool)
    masked_col_indices = col_indices[col_mask]
    return _column_mask_msa_indexed(msa, masked_col_indices, masking_type_distribution, static_mask_token,
                                    nonstatic_mask_tokens, start_token=start_token)


def _token_mask_msa(msa: torch.Tensor, p: float, masking_type_distribution: Distribution, static_mask_token: int, nonstatic_mask_tokens: List[int],
                    start_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masks out random tokens uniformly sampled from the given MSA, according to the given probability/ratio.
    Masked tokens are randomly replaced by tokens from a given token set.

    Args:
        msa (torch.Tensor): Tokenized MSA [E, L].
        p (float): Masking probability/ratio.
        masking_type_distribution (Distribution): Categorical distribution of conditional probabilities for each masking type.
        static_mask_token (int): Token that is used to replace masked tokens static masking.
        nonstatic_mask_tokens (List[int]): Tokens that are used to replace masked tokens randomly in nonstatic masking.
        start_token (bool, optional): Whether a start token is used, which is then precluded from masking. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L]
    """

    mask = torch.full(msa.size(), p)
    mask[:, :int(start_token)] = 0.
    mask = torch.bernoulli(mask).to(torch.bool)

    masking_type_sampling = masking_type_distribution.sample(mask.size())
    replace_mask = _get_replace_mask(mask, masking_type_sampling, static_mask_token, nonstatic_mask_tokens)

    masked = msa[mask]
    msa[mask] = replace_mask
    return msa, mask, masked


def _get_masking_fn(mode: str, start_token: bool) -> Callable[
        [torch.Tensor, float, Distribution, int, List[int]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ]:
    """
    Returns the masking function that corresponds to the given masking mode.

    Args:
        mode (str): Masking mode. Currently implemented: block-wise, column-wise, token-wise.
        start_token (bool): Whether a start token is used, which is then precluded from masking.

    Raises:
        ValueError: Unknown masking mode.

    Returns:
        Callable[[torch.Tensor, float, Distribution, int, List[int]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Masking function
        (tokenized MSA [E, L]; masking probability/ratio; distribution of conditional probabilities for each masking type;
        static-masking token; nonstatic-masking tokens -> masked MSA [E, L]; boolean masking mask [E, L]; masked tokens [~p*E*L])
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
        indices = torch.cat((torch.tensor([0]), (torch.randperm(max_nseqs-1)+1)[:nseqs-1]), dim=0)
        msa = MultipleSeqAlignment([msa[i.item()] for i in indices])
    return msa


def _subsample_diversity_maximizing(msa: MultipleSeqAlignment, nseqs: int, indices: torch.Tensor, contrastive: bool = False) -> MultipleSeqAlignment:
    """
    Subsamples sequences from the given MSA according to the diviserty maximization scheme.
    Requires pre-computed indices of most diverse sequences.

    Args:
        msa (MultipleSeqAlignment): Lettered MSA.
        nseqs (int): Number of sequences to be subsampled.
        indices (torch.Tensor): Indices of most diverse sequences.
        contrastive (bool, optional): Whether contrastive learning is active. Defaults to False.

    Returns:
        MultipleSeqAlignment: Subsampled, lettered MSA.
    """

    # diversity maximization should not be used in combination with contrastive
    assert not contrastive
    assert indices.shape[0] == nseqs
    assert len(msa) >= nseqs

    msa = MultipleSeqAlignment([msa[i.item()] for i in indices])
    return msa


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
    # TODO ensure this works, when there are not 2*nseqs sequences in the msa

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
