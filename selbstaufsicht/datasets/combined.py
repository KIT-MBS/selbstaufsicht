import torch

from .shrinked_force_permutations import ShrinkedForcePermutationsDataset


class CombinedDataset(ShrinkedForcePermutationsDataset):
    """
    Dataset that combines several datasets to a single one.
    Prerequisits: Passed datasets contain their data in the list attribute \"samples\" and inherit from \"ShrinkedForcePermutationsDataset\".
    Moreover, they are assumed to share the same \"transform\" attribute.
    """

    def __init__(self, *args: torch.utils.data.Dataset):
        assert len(args) > 0
        assert all(hasattr(dataset, 'samples') for dataset in args)
        assert all(isinstance(dataset, ShrinkedForcePermutationsDataset) for dataset in args)

        super().__init__()
        self.num_data_samples = args[0].num_data_samples
        self.jigsaw_force_permutations = args[0].jigsaw_force_permutations
        self.transform = args[0].transform
        self.samples = [sample for dataset in args for sample in dataset.samples]
        self._init_num_data_samples()
