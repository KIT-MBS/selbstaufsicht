import argparse
from datetime import datetime
from functools import partial
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from selbstaufsicht.utils import data_loader_worker_init, lehmer_encode, perm_gram_matrix, embed_finite_metric_space
from selbstaufsicht import models
from selbstaufsicht import datasets
from selbstaufsicht.models.self_supervised.msa.utils import get_tasks, get_downstream_transforms, MSACollator


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Training Script')
    # Architecture
    parser.add_argument('--checkpoint', type=str, help="Path to pre-trained model checkpoint")
    parser.add_argument('--num-blocks', default=6, type=int, help="Number of consecutive Transmorpher blocks")
    parser.add_argument('--feature-dim-head', default=64, type=int, help="Size of the feature dimension per Transmorpher head")
    parser.add_argument('--num-heads', default=12, type=int, help="Number of parallel Transmorpher heads")
    parser.add_argument("--disable-emb-grad-freq-scale", action='store_true', help="If set, this will scale gradients by the inverse of frequency of the words in the mini-batch")
    # Dataset
    parser.add_argument('--dataset', default='xfam', type=str, help="Used dataset: xfam, zwd, combined, dummy")
    parser.add_argument('--num-data-samples', default=-1, type=int, help="Number of used samples from dataset. Non-positive numbers refer to using all data.")
    parser.add_argument('--xfam-version', default='14.6', type=str, help="Xfam dataset version")
    parser.add_argument('--xfam-mode', default='seed', type=str, help="Xfam dataset mode: seed, full, or enhanced")
    # Training process
    parser.add_argument('--num-epochs', default=200, type=int, help="Number of training epochs")
    parser.add_argument('--batch-size', default=16, type=int, help="Batch size (local in case of multi-gpu training)")
    parser.add_argument('--validation-size', default=100, type=float, help="Either relative (if in range 0...1) or absolute (if integer) size of the validation dataset split")
    parser.add_argument('--learning-rate', default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument('--learning-rate-warmup', default=200, type=int, help="Warmup parameter for inverse square root rule of learning rate scheduling")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout probability")
    parser.add_argument('--precision', default=32, type=int, help="Precision used for computations")
    parser.add_argument('--attn-chunk-size', default=0, type=int, help="Chunk size in attention computation. Chunking causes sequential computation, which increases training time, but decreases memory pressure. Sizes below one activate the non-chunking implementation.")
    parser.add_argument('--disable-progress-bar', action='store_true', help="disables the training progress bar")
    parser.add_argument('--disable-shuffle', action='store_true', help="disables the dataset shuffling")
    parser.add_argument('--disable-random-split', action='store_true', help="disables the random dataset split")
    parser.add_argument('--rng-seed', default=42, type=int, help="Random number generator seed")
    # Data parallelism
    parser.add_argument('--num-gpus', default=-1, type=int, help="Number of GPUs per node. -1 refers to using all available GPUs. 0 refers to using the CPU.")
    parser.add_argument('--num-nodes', default=1, type=int, help="Number of nodes")
    parser.add_argument('--num-workers', default=4, type=int, help="Number of data loader worker processes")
    # Upstream tasks
    parser.add_argument('--task-inpainting', action='store_true', help="Activates the inpainting task")
    parser.add_argument('--task-jigsaw', action='store_true', help="Activates the jigsaw task")
    parser.add_argument('--task-contrastive', action='store_true', help="Activates the contrastive task")
    parser.add_argument('--task-jigsaw-boot', action='store_true', help="Activates the contrastive task")

    # Upstream task configuration
    parser.add_argument('--subsampling-depth', default=100, type=int, help="Number of subsampled sequences")
    parser.add_argument('--subsampling-mode', default='uniform', type=str, help="Subsampling mode: uniform, diversity, fixed")
    parser.add_argument('--cropping-size', default=400, type=int, help="Maximum uncropped sequence length")
    parser.add_argument('--cropping-mode', default='random-dependent', type=str, help="Cropping mode: random-dependent, random-independent, fixed")
    parser.add_argument('--inpainting-masking-type', default='token', type=str, help="MSA masking type in the inpainting task")
    parser.add_argument('--inpainting-masking-p', default=0.15, type=float, help="MSA masking ratio in the inpainting task")
    parser.add_argument('--inpainting-masking-p-static', default=0.8, type=float, help="Conditional probability that a token, if masked in the inpainting task, is replaced by a special masking token.")
    parser.add_argument('--inpainting-masking-p-nonstatic', default=0.1, type=float, help="Conditional probability that a token, if masked in the inpainting task, is replaced by a randomly drawn regular token.")
    parser.add_argument('--inpainting-masking-p-unchanged', default=0.1, type=float, help="Conditional probability that a token, if masked in the inpainting task, is not replaced.")
    parser.add_argument('--inpainting-loss-weight', default=1., type=float, help="Relative task loss weight. Is normalized before use.")
    parser.add_argument('--jigsaw-partitions', default=3, type=int, help="Number of partitions in the jigsaw task")
    parser.add_argument('--jigsaw-permutations', default=6, type=int, help="Number of permutations in the jigsaw task")
    parser.add_argument('--jigsaw-force-permutations', default=0, type=int,
                        help="""Duplicates the number of used data samples times the specified number in the jigsaw task,
                        where each duplicate is labeled with a different permutation in numerical order. Value 0 disables this mechanism."""
                        )
    parser.add_argument('--jigsaw-nonlinear', action='store_true', help="Uses a non-linear projection head for the jigsaw task.")
    parser.add_argument('--jigsaw-disable-delimiter', action='store_true', help="Disables delimiter token between partitions in the jigsaw task.")
    parser.add_argument('--jigsaw-euclid-emb', action='store_true', help="Uses an euclidean embedding of the discrete permutation metric for the jigsaw task.")
    parser.add_argument('--jigsaw-loss-weight', default=1., type=float, help="Relative task loss weight. Is normalized before use.")
    parser.add_argument('--contrastive-temperature', default=100., type=float, help="SimCLR temperature in the contrastive task")
    parser.add_argument('--contrastive-loss-weight', default=1., type=float, help="Relative task loss weight. Is normalized before use.")
    
    parser.add_argument('--jigsaw-boot-loss-weight', default=1., type=float, help="Relative task loss weight. Is normalized before use.")
    parser.add_argument('--jigsaw-boot-ratio',default=0.5,type=float,help="How many sequences from MSA to be bootstrapped")
    parser.add_argument('--boot-per-token',action='store_true',help="Per token loss")
    parser.add_argument('--boot-same',action='store_true',help="Compute per token loss between the replaced sequence and the new one")
    parser.add_argument('--frozen', action='store_true',help='applies the same permutation to all seqs in the MSA')


    # Logging
    parser.add_argument('--log-every', default=50, type=int, help='how often to add logging rows(does not write to disk)')
    parser.add_argument('--log-dir', default='lightning_logs/', type=str, help='Logging directory. Default: \"lightning_logs/\"')
    parser.add_argument('--log-exp-name', default='', type=str, help='Logging experiment name. If empty, this structure level is omitted. Default: \"\"')
    parser.add_argument('--log-run-name', default='%d_%m_%Y__%H_%M_%S', type=str,
                        help='Logging run name. Supports 1989 C standard datetime codes. Default: \"%%d_%%m_%%Y__%%H_%%M_%%S\"')

    args = parser.parse_args()
    dt_now = datetime.now()
    log_run_name = dt_now.strftime(args.log_run_name)

    tasks = []
    task_loss_weights = {}
    if args.task_inpainting:
        tasks.append("inpainting")
        task_loss_weights["inpainting"] = args.inpainting_loss_weight
    if args.task_jigsaw:
        tasks.append("jigsaw")
        task_loss_weights["jigsaw"] = args.jigsaw_loss_weight
    if args.task_contrastive:
        tasks.append("contrastive")
        task_loss_weights["contrastive"] = args.contrastive_loss_weight
    if args.task_jigsaw_boot:
        tasks.append("jigsaw_boot")
        task_loss_weights["jigsaw_boot"] = args.jigsaw_boot_loss_weight

    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    data_loader_rng = torch.Generator()
    data_loader_rng.manual_seed(args.rng_seed)

    num_gpus = args.num_gpus if args.num_gpus >= 0 else torch.cuda.device_count()
    if num_gpus * args.num_nodes > 1:
        dp_strategy = DDPPlugin(find_unused_parameters=False)
    else:
        dp_strategy = None

    if args.jigsaw_euclid_emb:
        # for num_partitions > 5, the euclidean embedding becomes highly inefficient
        if args.jigsaw_partitions > 5:
            raise ValueError("Euclidean embedding is too inefficient for num_partitions=%d!" % args.jigsaw_partitions)

        # check if embedding is available on disk
        emb_filename = 'jigsaw_euclid_emb_%d.pt' % args.jigsaw_partitions
        if os.path.isfile(emb_filename):
            # load embedding from disk
            jigsaw_euclid_emb = torch.load(emb_filename)
        else:
            if num_gpus > 0:
                raise ValueError("To compute the euclidean embedding, please pass num_gpus=0.")
            perm_indices = list(range(0, math.factorial(args.jigsaw_partitions)))
            perms = torch.stack([lehmer_encode(i, args.jigsaw_partitions) for i in perm_indices])
            d0 = perm_gram_matrix(perms)
            emb = embed_finite_metric_space(d0)
            torch.save(emb, emb_filename)
            print("Embedding computed and saved. Please restart the script.")
            exit(0)
    else:
        jigsaw_euclid_emb = None


    transform, task_heads, task_losses, train_metrics, val_metrics = get_tasks(tasks,
                                                                               args.feature_dim_head * args.num_heads,
                                                                               subsample_depth=args.subsampling_depth,
                                                                               subsample_mode=args.subsampling_mode,
                                                                               crop_size=args.cropping_size,
                                                                               crop_mode=args.cropping_mode,
                                                                               masking=args.inpainting_masking_type,
                                                                               p_mask=args.inpainting_masking_p,
                                                                               p_mask_static=args.inpainting_masking_p_static,
                                                                               p_mask_nonstatic=args.inpainting_masking_p_nonstatic,
                                                                               p_mask_unchanged=args.inpainting_masking_p_unchanged,
                                                                               jigsaw_partitions=args.jigsaw_partitions,
                                                                               jigsaw_classes=args.jigsaw_permutations,
                                                                               jigsaw_linear=not args.jigsaw_nonlinear,
                                                                               jigsaw_euclid_emb=jigsaw_euclid_emb,
                                                                               jigsaw_delimiter=not args.jigsaw_disable_delimiter,
                                                                               simclr_temperature=args.contrastive_temperature,
                                                                               jigsaw_boot_ratio=args.jigsaw_boot_ratio,
                                                                               per_token=args.boot_per_token,
                                                                               boot_same=args.boot_same,
                                                                               frozen=args.frozen)

    
    #print(transform)
    root = os.environ['DATA_PATH']
    dataset_name = args.dataset.lower()
    # NOTE MSA transformer: num_layers=12, d=768, num_heads=12, batch_size=512, lr=10**-4, **-2 lr schedule, 32 V100 GPUs for 100k updates, finetune for 25k more

    downstream_transform = get_downstream_transforms(subsample_depth=args.subsampling_depth, jigsaw_partitions=args.jigsaw_partitions)
    downstream_ds = datasets.CoCoNetDataset(root, 'train', transform=downstream_transform)
    test_ds = datasets.CoCoNetDataset(root, 'val', transform=downstream_transform)
    exclude_ids = downstream_ds.fam_ids + test_ds.fam_ids

    if dataset_name == 'xfam':
        dataset_path = os.path.join(root, 'Xfam')
        ds = datasets.XfamDataset(dataset_path, download=True, transform=transform, mode=args.xfam_mode, version=args.xfam_version, exclude_ids=exclude_ids)
    elif dataset_name == 'zwd':
        dataset_path = os.path.join(root, 'zwd')
        ds = datasets.ZwdDataset(dataset_path, transform=transform)
    elif dataset_name == 'combined':
        xfam_path = os.path.join(root, 'Xfam')
        zwd_path = os.path.join(root, 'zwd')
        xfam_ds = datasets.XfamDataset(xfam_path, download=True, transform=transform, mode=args.xfam_mode, version=args.xfam_version, exclude_ids=exclude_ids)
        zwd_ds = datasets.ZwdDataset(zwd_path, transform=transform)
        ds = datasets.CombinedDataset(xfam_ds, zwd_ds)
        del xfam_ds
        del zwd_ds
    elif dataset_name == 'dummy':
        ds = datasets.DummyDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: %s" % args.dataset)

    num_data_samples = args.num_data_samples if args.num_data_samples >= 0 else len(ds.samples)
    ds.num_data_samples = num_data_samples
    ds.jigsaw_force_permutations = args.jigsaw_force_permutations

    if type(args.validation_size) == int:
        validation_size = args.validation_size
    else:
        if 0 <= args.validation_size <= 1:
            validation_size = int(args.validation_size * num_data_samples)
        else:
            raise ValueError("Validation dataset size needs to be either in range 0...1 or an integer!")
    train_ds, val_ds = ds.split_train_val(validation_size, random=not args.disable_random_split)
    del ds

    #print(len(train_ds),"train_ds")
    #a,b=next(iter(train_ds))
    #print(a,b,"train_ds")

    train_dl = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          shuffle=not args.disable_shuffle,
                          collate_fn=MSACollator(train_ds.token_mapping['PADDING_TOKEN']),
                          num_workers=10,
                          worker_init_fn=partial(data_loader_worker_init, rng_seed=args.rng_seed),
                          generator=data_loader_rng, pin_memory=num_gpus > 0)
    val_dl = DataLoader(val_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=MSACollator(val_ds.token_mapping['PADDING_TOKEN']),
                        num_workers=10,
                        worker_init_fn=partial(data_loader_worker_init, rng_seed=args.rng_seed),
                        generator=data_loader_rng,
                        pin_memory=num_gpus > 0)

    #print(len(train_dl))

    # for j in range(0,3940):
    #     print(j)
    a,b=next(iter(train_dl))
    print(a,b)
    # #print(a['msa'][0])
    # #print(torch.eq(a['msa'][0][0],a['msa'][0][1]))
    #     ind_el=torch.LongTensor(np.zeros(a['msa'].shape[2]))
    #     ind_col=torch.LongTensor(np.arange(0,a['msa'].shape[2]))
    #     ind_rand=torch.ones(a['msa'].shape[1]).multinomial(num_samples=a['msa'].shape[2],replacement=True)
    # #print(ind_col," trala")
    #     indi=torch.stack((ind_el,ind_rand,ind_col),0)
    # #print(a['msa'].shape)
    # #aa=a['msa'][indi.chunk(chunks=a['msa'].shape[2], dim=0)]
    # #print(indi.chunk(chunks=a['msa'].shape[2], dim=0))
    #     aa=a['msa'][indi.chunk(chunks=a['msa'].shape[2], dim=0)]
    #     #print(aa.shape)
    #     #print(aa[0])
    #     #print(a['msa'][0][0])
    # #print(aa[0] in a['msa'][0])
    #     for i in range(0,a['msa'].shape[1]):
    #         #print(i,aa.shape[1])
    #         eqv=torch.eq(aa[0],a['msa'][0][i])
    #         false_values=eqv.masked_select(eqv == False)
    #         true_num = len(false_values)
    #         if true_num<1:
    #             print(i,"zero")
    #             break

    #i=0
    #while aa in a['msa'] and i<1000:
    #    ind_rand=torch.ones(a['msa'].shape[1]).multinomial(num_samples=a['msa'].shape[2],replacement=True)
    #    indi=torch.stack((ind_el,ind_rand,ind_col),0)
    #    aa=a['msa'][indi.chunk(chunks=a['msa'].shape[2], dim=0)]
    #    i=i+1
    #    print(i)

    #print(i)
    #p=torch.ones(a['msa'][:,:,2].shape)
    #p=torch.div(p,p.shape[1])
    #idx=p.multinomial(num_samples=3)
    #print(p)
    #print(idx)
    #print(a['msa'][:,idx,2],"tamtam")
    #print(a['msa'].size(),"blabla")
    #boo=torch.tensor()
    #from random import choices
    #print(choices(a['msa'][:,:,2],k=1))
    #print(a['msa'].shape)
    #print(a['padding_mask'].shape)
    #print(a['mask'].shape)
    #print(b['inpainting'].shape)
    #ttt=a['msa']
    #true_v=a['msa'].masked_select(a['mask'])
    #print(true_v)
#    for i in range(len(train_dl)):
#        a,b=next(iter(train_dl))
#        if a['msa'].shape[1]<20:
#            print(b)
            #print(a['msa'].shape)
    #a,b=next(iter(train_dl))
    #print(a['msa'])
    #print(a['msa'].shape)
    #a,b=next(iter(train_dl))
    #print(a['msa'])
    #print(a['msa'].shape)


    model = models.self_supervised.MSAModel(
    #models.self_supervised.MSAModel.load_from_checkpoint(
        #num_blocks=
        args.num_blocks,
        #num_heads=
        args.num_heads,
        #feature_dim_head=
        args.feature_dim_head,
        #checkpoint_path=args.checkpoint,
        task_heads=task_heads,
        task_losses=task_losses,
        task_loss_weights=task_loss_weights,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        alphabet_size=len(train_ds.token_mapping),
        padding_token=train_ds.token_mapping['PADDING_TOKEN'],
        lr=args.learning_rate,
        lr_warmup=args.learning_rate_warmup,
        dropout=args.dropout,
        emb_grad_freq_scale=not args.disable_emb_grad_freq_scale,
        h_params=args,
        attn_chunk_size=args.attn_chunk_size
    )
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_exp_name, version=log_run_name)
    trainer = Trainer(max_epochs=args.num_epochs,
                      gpus=args.num_gpus,
                      auto_select_gpus=num_gpus > 0,
                      num_nodes=args.num_nodes,
                      precision=args.precision,
                      strategy=dp_strategy,
                      enable_progress_bar=not args.disable_progress_bar,
                      log_every_n_steps=min(args.log_every, num_data_samples),
                      logger=tb_logger)
    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    main()
