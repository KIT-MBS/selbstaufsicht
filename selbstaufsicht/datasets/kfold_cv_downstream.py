from functools import partial
import os
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Subset, random_split
from typing import Union

from selbstaufsicht.datasets import CoCoNetDataset
from selbstaufsicht.utils import data_loader_worker_init


class KFoldCVDownstream():
    def __init__(self, transform, num_folds: int = 1, val_ratio: float = 0.1, batch_size: int = 1, shuffle: bool = True, rng_seed: int = 42,
                 discard_train_size_based=True, diversity_maximization=False, max_seq_len: int = 400, min_num_seq: int = 50) -> None:
        """
        Initializes K-fold cross validation for the downstream task.

        Args:
            transform: Data transforms
            num_folds (int, optional): Number of folds. Defaults to 1.
            val_ratio (float, optional): Validation dataset ratio in case of disabled cross validation (num_folds=1). Defaults to 0.1.
            batch_size (int, optional): Batch size. Defaults to 1.
            shuffle (bool, optional): Whether data is shuffled. Defaults to True.
            rng_seed (int, optional): DataLoader/Shuffling RNG seed. Defaults to 42.
            discard_train_size_based (bool, optional): Whether training data is discarded, if it contravenes size criteria. Defaults to True.
            diversity_maximization (bool, optional): Whether diversity maximization is used as subsampling strategy. Defaults to False.
            max_seq_len (int, optional): Maximum sequence length s.t. the MSA is not discarded. Defaults to 400.
            min_num_seq (int, optional): Minimum number of sequences s.t. the MSA is not discarded. Defaults to 50.
        """

        assert num_folds > 0
        assert 0 < val_ratio < 1

        self.num_folds = num_folds
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_seed = rng_seed

        # set rng seed
        self.data_loader_rng = torch.Generator()
        self.data_loader_rng.manual_seed(rng_seed)

        # load dataset
        self.root = os.environ['DATA_PATH']
        self.train_dataset = CoCoNetDataset(self.root, 'train', transform=transform, discard_train_size_based=discard_train_size_based,
                                            diversity_maximization=diversity_maximization, max_seq_len=max_seq_len, min_num_seq=min_num_seq)

        # setup splits
        if self.num_folds == 1:
            val_size = int(self.val_ratio * len(self.train_dataset))
            train_size = len(self.train_dataset) - val_size
            self.train_fold, self.val_fold = random_split(self.train_dataset, [train_size, val_size], self.data_loader_rng)
        else:
            self.splits = [split for split in KFold(self.num_folds, shuffle=self.shuffle, random_state=self.rng_seed).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        """
        Setups concrete cross validation fold.

        Args:
            fold_index (int): Fold index.
        """

        if self.num_folds >= 2:
            train_indices, val_indices = self.splits[fold_index]
            self.train_fold = Subset(self.train_dataset, train_indices)
            self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0,
                          worker_init_fn=partial(data_loader_worker_init, rng_seed=self.rng_seed),
                          generator=self.data_loader_rng, pin_memory=False)

    def val_dataloader(self) -> Union[DataLoader, None]:
        return DataLoader(self.val_fold, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          worker_init_fn=partial(data_loader_worker_init, rng_seed=self.rng_seed),
                          generator=self.data_loader_rng, pin_memory=False)
