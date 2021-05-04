import torch


# def get_transforms(pmask=0.3, gaps=True, contiguous=False):
#     ts = Compose([RelatedSequenceReconstructionTask(pmask, gaps, contiguous), SequencesToTensor()])
#     return ts


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RelatedSequenceReconstructionTask():
    """
    transform for the related sequence reconstruction task.
    It extracts two random sequences from the alignment, masks out positions in the target sequence to reconstruct
    """
    def __init__(self, pmask=0.3, gaps=True, contiguous=False):
        self.p = pmask
        if not gaps:
            raise NotImplementedError("This task is not yet implemented.")
        if contiguous:
            raise NotImplementedError("This task is not yet implemented.")

    def __call__(self, msa):
        n = len(msa)
        i = torch.randint(0, n, (1,)).item()
        j = torch.randint(0, n, (1,)).item()
        source = msa[i]
        target = msa[j]
        return source, target

class SequenceReconstructionTask():
    def __init__(self, pmask=0.3, gaps=True, contiguous=False):
        self.p = pmask
        if not gaps:
            raise NotImplementedError("This task is not yet implemented.")
        if contiguous:
            raise NotImplementedError("This task is not yet implemented.")
    def __call__(self, msa):
        n = len(msa)
        i = torch.randint(0, n, (1,)).item()
        s = msa[i]
        t = msa[i]

        s = [l if torch.rand((1,)).item() > self.p else letters[torch.randint(0, len(letters), (1,)).item()] for l in s]
        return (s, t)

# TODO handle msas and other needed constellations of sequences
class SequencesToTensor():
    def __init__(self):
        return

    def __call__(self, sequences):
        return sequence_to_tensor_(sequences[0]), sequence_to_tensor_(sequences[1])

letters = '-ACGUNYRWSMVBKD'
rna_letter_dict = {l: i for (i, l) in enumerate(letters)}

def sequence_to_tensor_(s):
    return torch.tensor([rna_letter_dict[l] for l in s], dtype=torch.long)

class SequenceToTensor():
    def __init__(self):
        return

    def __call__(self, s):
        return sequence_to_tensor_(s)
