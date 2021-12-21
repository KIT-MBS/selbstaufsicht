import torch
from typing import Tuple

from ..utils import rna2index


class ShrinkedForcePermutationsDataset(torch.utils.data.Dataset):
    """
    Dataset that supports dataset shrinking and forced jigsaw permutations by MSA duplications.
    """

    def __init__(self, num_data_samples=0, jigsaw_force_permutations=0, samples=[], transform=None):
        self.num_data_samples = num_data_samples
        self.jigsaw_force_permutations = jigsaw_force_permutations
        self.samples = samples
        self.transform = transform
        self.token_mapping = rna2index

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

    def split_train_val(self, validation_size: int, random: bool = True) -> Tuple['ShrinkedForcePermutationsDataset', 'ShrinkedForcePermutationsDataset']:
        """
        Splits the given dataset into a training and a validation dataset.

        Args:
            validation_size (int): Number of samples to be contained by the validation dataset.
            random (bool, optional): Whether split is random. Defaults to True.

        Returns:
            Tuple[ShrinkedForcePermutationsDataset, ShrinkedForcePermutationsDataset]: Training dataset, validation dataset
        """

        assert validation_size < self.num_data_samples

        training_dataset = ShrinkedForcePermutationsDataset(self.num_data_samples - validation_size, self.jigsaw_force_permutations, transform=self.transform)
        validation_dataset = ShrinkedForcePermutationsDataset(validation_size, self.jigsaw_force_permutations, transform=self.transform)

        if random:
            # create a set of <validation_size> unique numbers in the range 0..<num_data_samples>-1
            validation_indices = set(torch.randperm(self.num_data_samples, dtype=int)[:validation_size].tolist())

            for idx in range(self.num_data_samples):
                if idx in validation_indices:
                    validation_dataset.samples.append(self.samples[idx])
                else:
                    training_dataset.samples.append(self.samples[idx])

            # <num_data_samples> might be smaller than the original dataset size
            # -> add remaining samples to training dataset (they will not be used, unless <num_data_samples> is increased)
            training_dataset.samples += self.samples[self.num_data_samples:]
        else:
            # first samples are used for validation, last ones for training
            validation_dataset.samples += self.samples[:validation_size]
            training_dataset.samples += self.samples[validation_size:]

        return training_dataset, validation_dataset
