import torch

class ShrinkedForcePermutationsDataset(torch.utils.data.Dataset):
    """
    Dataset that supports dataset shrinking and forced jigsaw permutations by MSA duplications.
    """

    def __init__(self):
        self.num_data_samples = 0
        self.jigsaw_force_permutations = 0

    def __getitem__(self, idx):
        if self.transform is not None:
            if self.jigsaw_force_permutations:
                # data-sample-major order (batches of data samples, which are labeled with several permutations)
                real_sample_idx = idx // self.jigsaw_force_permutations
                permutation_idx = idx % self.jigsaw_force_permutations
                sample = self.samples[real_sample_idx]
                num_seq = len(sample)
                labels = {'jigsaw': torch.full((num_seq,), permutation_idx, dtype=torch.int64)}
            else:
                sample = self.samples[idx]
                labels = {}
            return self.transform({'msa': sample}, labels)
        return self.samples[idx]

    def __len__(self):
        if self.jigsaw_force_permutations:
            return min(len(self.samples), self.num_data_samples) * self.jigsaw_force_permutations
        else:
            return min(len(self.samples), self.num_data_samples)