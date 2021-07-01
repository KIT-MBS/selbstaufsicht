import torch

# ambuguous RNA letters: GAUCRYWSMKHBVDN
rna_letters = [letter for letter in '-GAUCRYWSMKHBVDN']
rna_to_index = {letter: index for index, letter in enumerate(rna_letters)}

rna_to_tensor_dict = {
        '-': torch.tensor([1., 0., 0., 0., 0., 0.]),
        'G': torch.tensor([0., 1., 0., 0., 0., 0.]),
        'A': torch.tensor([0., 0., 1., 0., 0., 0.]),
        'U': torch.tensor([0., 0., 0., 1., 0., 0.]),
        'C': torch.tensor([0., 0., 0., 0., 1., 0.]),
        'R': torch.tensor([0., 1., 1., 0., 0., 0.])/2.,
        'Y': torch.tensor([0., 0., 0., 1., 1., 0.])/2.,
        'W': torch.tensor([0., 0., 1., 1., 0., 0.])/2.,
        'S': torch.tensor([0., 1., 0., 0., 1., 0.])/2.,
        'M': torch.tensor([0., 0., 1., 0., 1., 0.])/2.,
        'K': torch.tensor([0., 1., 0., 1., 0., 0.])/2.,
        'H': torch.tensor([0., 0., 1., 1., 1., 0.])/3.,
        'B': torch.tensor([0., 1., 0., 1., 1., 0.])/3.,
        'V': torch.tensor([0., 1., 1., 0., 1., 0.])/3.,
        'D': torch.tensor([0., 1., 1., 1., 0., 0.])/3.,
        'N': torch.tensor([0., 1., 1., 1., 1., 0.])/4.,
        }

# TODO bucketing
# TODO more efficient positional encoding?
# TODO reorganize
# TODO double check
# TODO optimize: best batch layout, better tensorize
# TODO maxlen as parameter
def collate_msas_explicit_position(msas):
    seqlens = [len(msa[0]) for msa in msas]
    maxlen = 1000
    B = len(msas) # batch size
    S = max(len(msa) for msa in msas) # number of sequences in alignment
    L = max(seqlens) # length of sequences in alignment
    D = 8 # gap + 4 letters + mask token + 2 dims for position

    peindex = torch.arange(0, L)

    batch = torch.zeros((B, S, L, D), dtype=torch.float)

    for i, msa in enumerate(msas):
        for s in range(len(msa)):
            for l in range(len(msa[0])):
                batch[i, s, l, 0:6] = rna_to_tensor_dict[msa[s, l]][:]
                batch[i, s, l, 6] = peindex[l] / maxlen
                batch[i, s, l, 7] = peindex[l] / len(msa[s])

    return (batch, seqlens)

def lehmer_encode(i, n):
    """
    Encodes an integer i in the interval [0,n!-1] as a permutation of (0,1,2,...,n-1).

    Example:
        >>> lehmer_encode(10,7)
        tensor([0, 1, 2, 4, 6, 3, 5], dtype=torch.int32)
    """
    pos = torch.empty((n-1,),dtype=torch.int32)
    for j in range(2,n+1):
        ii = i // j
        pos[n-j] = i - ii * j
        i = ii

    assert(i == 0)

    init = torch.arange(n,dtype=torch.int32)
    ret = torch.empty((n,),dtype=torch.int32)
    for j in range(n-1):
        jj = 0
        for k in range(n):
            if init[k] == -1:
                continue
            if jj == pos[j]:
                ret[j] = k # should be the same as init[k]
                init[k] = -1
                break
            jj += 1

    for k in range(n):
        if init[k] == -1:
            continue
        ret[n-1] = k

    return ret
