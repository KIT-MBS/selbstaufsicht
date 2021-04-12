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
