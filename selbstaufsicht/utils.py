import torch

# ambuguous RNA letters: GAUCRYWSMKHBVDN
mask_token = '*'
delimiter_token = '|'
rna_letters = [letter for letter in '-GAUCRYWSMKHBVDN']
rna_letters += [mask_token, delimiter_token]

rna2index = {letter: index for index, letter in enumerate(rna_letters)}

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


def lehmer_encode(i, n):
    """
    Encodes an integer i in the interval [0,n!-1] as a permutation of (0,1,2,...,n-1).

    Example:
        >>> lehmer_encode(10,7)
        tensor([0, 1, 2, 4, 6, 3, 5], dtype=torch.int32)
    """
    pos = torch.empty((n-1,), dtype=torch.int32)
    for j in range(2, n+1):
        ii = i // j
        pos[n-j] = i - ii * j
        i = ii

    assert(i == 0)

    init = torch.arange(n, dtype=torch.int32)
    ret = torch.empty((n,), dtype=torch.int32)
    for j in range(n-1):
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
        ret[n-1] = k

    return ret


# TODO expect a target dict
def pad_collate_fn(batch):
    raise
    return batch
