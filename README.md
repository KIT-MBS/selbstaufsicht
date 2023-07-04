# BARNACLE

RNA Contact prediction with deep learning

## Running scripts

- Scripts for running different methods are in the `scripts` folder
  - NOTE: these scripts will likely take a very long time on a desktop computer without a datacenter grade GPU.
  - train.py runs the self-supervised pre-training. On four A100 GPUs it runs about 48 hours.
  - train_downstream.py runs the downstream regression training. Runs less than one hour.
  - xgboost_downstream.py runs the downstream xgboost training. Runs less than 15 minutes.
- Source code is in `selbstaufsicht`

## Installation guide

- Setup venv: `python -m venv venv_name`
- clone this repo: `git clone .....`
- change directory into the newly cloned repository
- Install dependencies from this repo: `pip install -e .`

Installation shouldnt take more than 5 minutes (dependent on internet connection speed).
Please note, that a GPU is highly recommended for running the scripts within. For more information, see the hardware requirements below.

## Software dependencies
- reproduced from setup.py

```
python >= 3.9
protobuf < 4.21.0  # higher versions lead to errors with pytorch-lightning (https://github.com/PyTorchLightning/pytorch-lightning/issues/13159)
torch >= 1.10, <= 1.13.1  # range supported by pytorch-lightning 1.8.x
torchvision
tensorboard
tqdm
biopython
axial_attention #get rid of this and replace with optimized custom stuff
pytorch-lightning == 1.8.6  # lightning-bolts breaks for higher versions
lightning-bolts
cvxpy
seaborn
matplotlib
scikit-learn
pandas
xgboost
```

## Singularity Container
We provide a singularity container for those who prefer to use that [here](https://zenodo.org/record/7858123).

## Hardware requiremenst
- it would be best to use multiple GPUs for training. if using one GPU alone, it may take a long time
- It is further recommended to have a GPU with at 40 GB of VRAM or more to avoid OOM errors when training
- Running the trained models without validation requires less VRAM

## Workflows
### Training
Export a `DATA_PATH` environment variable, where the training and test data will be stored.

For self-supervised pre-training, use one of:
```
python train.py --dataset combined --batch-size 1 --num-workers 8 --num-epochs 10000 --learning-rate 0.00003 --learning-rate-warmup 400 --dropout 0.3 --subsampling-depth 50 --precision 16 --disable-emb-grad-freq-scale --disable-progress-bar --log-every 500 --num-gpus 4 --task-inpainting --num-blocks 10 --num-heads 12 --feature-dim-head 64 --inpainting-masking-p-static 0.0 --inpainting-masking-p-nonstatic 1.0 --inpainting-masking-p-unchanged 0.0 --log-run-name inpainting


python train.py --dataset combined --batch-size 1 --num-workers 8 --num-epochs 10000 --learning-rate 0.00003 --learning-rate-warmup 400 --dropout 0.3 --subsampling-depth 50 --precision 16 --disable-emb-grad-freq-scale --disable-progress-bar --log-every 500 --num-gpus 4 --task-inpainting --task-jigsaw --num-blocks 10 --num-heads 12 --feature-dim-head 64 --inpainting-masking-p-static 0.0 --inpainting-masking-p-nonstatic 1.0 --inpainting-masking-p-unchanged 0.0 --log-run-name jigsaw

python train.py --dataset combined --batch-size 1 --num-workers 8 --num-epochs 10000 --learning-rate 0.00003 --learning-rate-warmup 400 --dropout 0.3 --subsampling-depth 50 --precision 16 --disable-emb-grad-freq-scale --disable-progress-bar --log-every 500 --num-gpus 4 --task-inpainting --task-contrastive --num-blocks 10 --num-heads 12 --feature-dim-head 64 --inpainting-masking-p-static 0.0 --inpainting-masking-p-nonstatic 1.0 --inpainting-masking-p-unchanged 0.0 --log-run-name contrastive_100 --contrastive-temperature 100.0

python train.py --dataset combined --batch-size 1 --num-workers 8 --num-epochs 10000 --learning-rate 0.00003 --learning-rate-warmup 400 --dropout 0.3 --subsampling-depth 50 --precision 16 --disable-emb-grad-freq-scale --disable-progress-bar --log-every 500 --num-gpus 4 --task-inpainting --task-jigsaw-boot --boot-per-token --num-blocks 10 --num-heads 12 --feature-dim-head 64 --inpainting-masking-p-static 0.0 --inpainting-masking-p-nonstatic 1.0 --inpainting-masking-p-unchanged 0.0 --log-run-name bootstrap

```

For regression downstream training, use;
```
python train_downstream.py --loss-contact-weight 0.95 --loss focal --precision 16 --num-epochs 10000 --learning-rate-warmup 0 --dropout 0 --cv-num-folds 1 --validation-ratio 0.2 --subsampling-mode diversity --checkpoint <backbone_checkpoint_path> --log-run-name <run name> --disable-progress-bar (--freeze-backbone)
```

For XGBoost downstream training, use:
```
python xgboost_downstream.py --subsampling-mode diversity --num-round 300 --learning-rate 1.0 --gamma 0.7 --max-depth 16 --min-child-weight 1 --colsample-bytree 0.7 --colsample-bylevel 0.7 --xgb-subsampling-rate 0.9 --xgb-subsampling-mode g    radient_based --dart-dropout 0.1 --top-l-prec-coeff 1.0 --cv-num-folds 1  --checkpoint <checkpoint_path> --log-run-name <run_name> --disable-progress-bar --monitor-metric <['toplprec', 'toplprecpos', 'f1', 'matthews']>
```
The checkpoint path given is the one to the pre-trained model checkpoint or to the one finetuned without --freeze-backbone

### Testing
For regression:
```
python test_downstream.py --subsampling-mode diversity --checkpoint <checkpoint_path> --disable-progress-bar

```
For XGBoost:
```
python test_xgboost_downstream.py --subsampling-mode diversity --num-k 1 --min-k 1  --checkpoint <checkpoint_path> --xgboost-checkpoint <xgboost_checkpoint>

```
The checkpoint path given is the on to the final downstream model checkpoint.
XGBoost requires one torch model checkpoint and one XGBoost model checkpoint.
