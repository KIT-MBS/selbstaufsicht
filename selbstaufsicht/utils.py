import random
import numpy as np
import torch

# TODO token mapping should be part of dataset?
# ambuguous RNA letters: GAUCRYWSMKHBVDN
rna_letters = [letter for letter in '-.GAUCRYWSMKHBVDN']
# TODO protein_letters = [letter for letter in '']

rna2index = {letter: index for index, letter in enumerate(rna_letters)}
rna2index['START_TOKEN'] = len(rna2index)
rna2index['DELIMITER_TOKEN'] = len(rna2index)
# NOTE these two have to be last, since they can not be inpainted
rna2index['MASK_TOKEN'] = len(rna2index)
rna2index['PADDING_TOKEN'] = len(rna2index)

# rnaletter2tensor_encoded_ambiguity_dict = {
#         '-': torch.tensor([1., 0., 0., 0., 0., 0.]),
#         'G': torch.tensor([0., 1., 0., 0., 0., 0.]),
#         'A': torch.tensor([0., 0., 1., 0., 0., 0.]),
#         'U': torch.tensor([0., 0., 0., 1., 0., 0.]),
#         'C': torch.tensor([0., 0., 0., 0., 1., 0.]),
#         'R': torch.tensor([0., 1., 1., 0., 0., 0.])/2.,
#         'Y': torch.tensor([0., 0., 0., 1., 1., 0.])/2.,
#         'W': torch.tensor([0., 0., 1., 1., 0., 0.])/2.,
#         'S': torch.tensor([0., 1., 0., 0., 1., 0.])/2.,
#         'M': torch.tensor([0., 0., 1., 0., 1., 0.])/2.,
#         'K': torch.tensor([0., 1., 0., 1., 0., 0.])/2.,
#         'H': torch.tensor([0., 0., 1., 1., 1., 0.])/3.,
#         'B': torch.tensor([0., 1., 0., 1., 1., 0.])/3.,
#         'V': torch.tensor([0., 1., 1., 0., 1., 0.])/3.,
#         'D': torch.tensor([0., 1., 1., 1., 0., 0.])/3.,
#         'N': torch.tensor([0., 1., 1., 1., 1., 0.])/4.,
#         mask_token: torch.tensor(),  # TODO
#         delimiter_token: torch.tensor()  # TODO
#         }


def lehmer_encode(i: int, n: int) -> torch.Tensor:
    """
    Encodes an integer i in the interval [0,n!-1] as a permutation of (0,1,2,...,n-1).

    Args:
        i (int): Lehmer index.
        n (int): Number of elements in the sequence to be permuted.
        
    Example:
        >>> lehmer_encode(10,7)
        tensor([0, 1, 2, 4, 6, 3, 5], dtype=torch.int32)

    Returns:
        torch.Tensor: Permutation.
    """

    pos = torch.empty((n - 1,), dtype=torch.int32)
    for j in range(2, n + 1):
        ii = i // j
        pos[n - j] = i - ii * j
        i = ii

    assert(i == 0)

    init = torch.arange(n, dtype=torch.int32)
    ret = torch.empty((n,), dtype=torch.int32)
    for j in range(n - 1):
        jj = 0
        for k in range(n):
            if init[k] == -1:
                continue
            if jj == pos[j]:
                ret[j] = k  # should be the same as init[k]
                init[k] = -1
                break
            jj += 1

    for k in range(n):
        if init[k] == -1:
            continue
        ret[n - 1] = k

    return ret


def data_loader_worker_init(worker_id: int, rng_seed: int) -> None:
    """
    Initialization method for data loader workers, which fixes the random number generator seed.

    Args:
        worker_id (int): Worker ID.
        rng_seed (int): Random number generator seed.
    """
    
    np.random.seed(rng_seed)
    random.seed(rng_seed)