# selbstaufsicht
Self-Supervised Learning for Applications in Bioinformatics.

TODO:
- new `subsampling_mode`: fixed (take first n sequences)
- adapt `num_gpu` in trainung script: `num_gpu` -> cpu utilization
- set new default values in training script
- test duplicated MSA, fixed subsampling, different permutations
- make new `DummyDataset` variants (more complex token patterns)
- test utilization of delimiter tokens for classification in the `JigsawHead` as well
- test non-linear architecture for `JigsawHead`
- parameterize tasks
- train test split: large unlabeled train (~4000), small unlabeled val (~100), small labeled train (~10), large labeled val (~100)
- NAS
- downstream validation
- separate, but weight-shared tasks
- masking: combined masking modes (optional)
- write paper
- submit paper
