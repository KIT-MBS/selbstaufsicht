# selbstaufsicht
Self-Supervised Learning for Applications in Bioinformatics.

TODO:
- ~~try focal loss or dice loss~~  (focal loss slightly better than pure cross entropy, dice loss worse)
- ~~implement cross validation akin to coconet~~
- ~~try LBFGS with frozen backbone~~  (no positive effects/job freezes randomly with higher lr)
- investigate dataset/distribution/generalization issues
    - ~~look at individual predictions in both cases~~ (discarded hammerhead ribozyme, MSAs smaller than 50/longer than 400)
    - visualize topL predictions on the reference structure
- implement pre-computed diversity-maximizing subsampling
- per sequence contrastive
- separate, but weight-shared tasks
- clean up dataset stuff
- downstream optimizer
- hyperparameter
    - SGD ADAM
- ablation study:
    - upstream optimizer
    - backbone complexity
    - downstream depth
    - downstream optimizer
    - exlude low meff MSAs from upstream training
- write paper
- submit paper
