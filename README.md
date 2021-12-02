# selbstaufsicht
Self-Supervised Learning for Applications in Bioinformatics.

TODO:
- dummy train dataset
- refactor transformations: dataset returns tuple of dicts, transforms only expect and manipulate content of dicts, typehinting
- ~~test jigsaw with delimiter~~ (DONE #8)
- investigate whether embedding should be more complex
- ~~train script parameter: seed, dataset selector, dataset modes~~ (DONE #9)
- ~~reproducibility: script seed, dataloader worker init~~ (DONE #9)
- masking: combined masking modes
- jigsaw: testing, explore querying only start token, explore jigsawing every individual sequence in the alignment
- train test split: large unlabeled train (~4000), small unlabeled val (~100), small labeled train (~10), large labeled val (~100)
- parameterize tasks
- NAS
- write paper
- submit paper
