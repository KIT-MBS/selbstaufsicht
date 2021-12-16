# selbstaufsicht
Self-Supervised Learning for Applications in Bioinformatics.

TODO:
- concatenate zwd and rfam datasets, such that force-permutations is still working [DONE [#18](https://github.com/KIT-MBS/selbstaufsicht/pull/18)]
- train test split upstream: large unlabeled train (~4000), small unlabeled val (~100) [DONE [#19](https://github.com/KIT-MBS/selbstaufsicht/pull/19)]
- task loss weights (1:1, 1:1.5, ..., 1:4 with Jigsaw:Inpainting)
- run one training into walltime limit (best task-weights, dropout=0.1, 0.2, 0.3)
- DROPOUT/REGULARIZATION
- per sequence contrastive
- train test split downstream: small labeled train (~10), large labeled val (~100)
- downstream validation
- parameterize tasks
- NAS
- separate, but weight-shared tasks
- test utilization of delimiter tokens for classification in the `JigsawHead` as well (optional)
- masking: combined masking modes (optional)
- write paper
- submit paper
