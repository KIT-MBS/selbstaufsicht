import torch


# ambuguous RNA letters: GAUCRYWSMKHBVDN
rna_to_tensor_dict = {
        '-': torch.tensor([0., 0., 0., 0.]),
        '.': torch.tensor([0., 0., 0., 0.]),
        'G': torch.tensor([1., 0., 0., 0.]),
        'A': torch.tensor([0., 1., 0., 0.]),
        'U': torch.tensor([0., 0., 1., 0.]),
        'C': torch.tensor([0., 0., 0., 1.]),
        'R': torch.tensor([1., 1., 0., 0.])/2.,
        'Y': torch.tensor([0., 0., 1., 1.])/2.,
        'W': torch.tensor([0., 1., 1., 0.])/2.,
        'S': torch.tensor([1., 0., 0., 1.])/2.,
        'M': torch.tensor([0., 1., 0., 1.])/2.,
        'K': torch.tensor([1., 0., 1., 0.])/2.,
        'H': torch.tensor([0., 1., 1., 1.])/3.,
        'B': torch.tensor([1., 0., 1., 1.])/3.,
        'V': torch.tensor([1., 1., 0., 1.])/3.,
        'D': torch.tensor([1., 1., 1., 0.])/3.,
        'N': torch.tensor([1., 1., 1., 1.])/4.,
        }

# TODO reorganize
# TODO optimize
def collate_msas(msas):
    B = len(msas)
    S = max([len(msa) for msa in msas])
    L = max([msa.get_alignment_length() for msa in msas])
    D = 4

    batch = torch.zeros((S, L, B, D), dtype=torch.float)

    for i, msa in enumerate(msas):
        for s in range(len(msa)):
            for l in range(msa.get_alignment_length()):
                batch[s, l, i, ...] = rna_to_tensor_dict[msa[s, l]][...]

    batch = batch.reshape(S*L, B, D)
    return (batch, torch.Tensor(batch))
