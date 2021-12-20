import torch

from .shrinked_force_permutations import ShrinkedForcePermutationsDataset

class CombinedDataset(ShrinkedForcePermutationsDataset):
    """
    Dataset that combines several datasets to a single one.
    Prerequisits: Passed datasets contain their data in the list attribute \"samples\" and inherit from \"ShrinkedForcePermutationsDataset\".
    Moreover, they are assumed to share the same \"token_mapping\" and \"transform\" attributes.
    """

    def __init__(self, *args: torch.utils.data.Dataset):
        assert len(args) > 0
        assert all(hasattr(dataset, 'samples') for dataset in args)
        assert all(isinstance(dataset, ShrinkedForcePermutationsDataset) for dataset in args)
        
        self.num_data_samples = args[0].num_data_samples
        self.jigsaw_force_permutations = args[0].jigsaw_force_permutations
        self.token_mapping = args[0].token_mapping
        self.transform = args[0].transform
        self.samples = [sample for dataset in args for sample in dataset.samples]
