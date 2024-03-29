import os
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import xgboost as xgb
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    mean_squared_error,
    pearson_corrcoef,
    spearman_corrcoef,
)

from selbstaufsicht import datasets, models
from selbstaufsicht.models.self_supervised.msa.utils import (
    get_downstream_transforms,
    get_tasks,
)
from selbstaufsicht.utils import data_loader_worker_init


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies sigmoid function.

    Args:
        x (np.ndarray): Input data.

    Returns:
        np.ndarray: Output data.
    """

    return 1 / (1 + np.exp(-x))


def xgb_topkLPrec_var_k(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray, L_mapping: np.ndarray, k: np.ndarray, relative_k: bool = True, treat_all_preds_positive: bool = True) -> Union[Dict[float, np.ndarray], Dict[int, Dict[int, float]]]:
    """
    Custom XGBoost Metric for top-L-precision with support for various k (relative / absolute number of contacts).

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        k (np.ndarray): Coefficients / counts k that are used in computing the top-(k*L)-precision / top-k-precision [num_k, num_msa].
        relative_k (bool): Whether k contains relative coefficients or absolute counts. Defaults to True.
        treat_all_preds_positive (bool, optional): Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper. Defaults to True.

    Returns:
        Union[Dict[float, np.ndarray], Dict[int, Dict[int, float]]]: Metric values per k (relative) / metric values per k per MSA (absolute).
    """

    y = dtest.get_label()  # [B]

    msa_indices = np.unique(msa_mapping)

    top_l_prec_dict = dict()

    # for each MSA, find top-L and compute true/false positives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]

        for k_idx in range(len(k)):
            k_ = k[k_idx][msa_idx]
            L = L_mapping[msa_idx]
            if relative_k:
                k_L = min(max(1, int(k_*L)), len(y_))
            else:
                k_L = min(max(1, int(k_)), len(y_))
            stop_flag = k_L == len(y_)
            L_idx = np.argpartition(preds_, -k_L)[-k_L:]  # [k*L]

            preds__ = np.round(sigmoid(preds_[L_idx]))
            y__ = y_[L_idx]

            if treat_all_preds_positive:
                tp = sum(y__ == 1)
                fp = sum(y__ == 0)
            else:
                tp = sum(np.logical_and(preds__ == 1, y__ == 1))
                fp = sum(np.logical_and(preds__ == 1, y__ == 0))

            top_l_prec = float(tp) / (tp + fp)

            if k_ not in top_l_prec_dict:
                top_l_prec_dict[k_] = dict()

            top_l_prec_dict[k_][msa_idx] = top_l_prec

            if stop_flag and relative_k:
                break

    if relative_k:
        return {k: np.array(list(v.values())) for k, v in top_l_prec_dict.items()}
    else:
        unsorted = {k2: {k1: top_l_prec_dict[k1][k2] for k1 in top_l_prec_dict if k2 in top_l_prec_dict[k1]} for k2 in msa_indices}
        return {k2: OrderedDict(sorted(unsorted[k2].items(), key=lambda t: t[0])) for k2 in msa_indices}


def xgb_topkLPrec(preds: np.ndarray, dmat: xgb.DMatrix, msa_mapping: np.ndarray, L_mapping: np.ndarray, k: float = 1., treat_all_preds_positive: bool = False) -> float:
    """
    Custom XGBoost Metric for top-L-precision.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dmat (xgb.DMatrix): Input data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        k (float, optional): Coefficient k that is used in computing the top-(k*L)-precision. Defaults to 1.
        treat_all_preds_positive (bool, optional): Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper. Defaults to False.

    Returns:
        float: Metric value.
    """

    y = dmat.get_label()  # [B]

    msa_indices = np.unique(msa_mapping)
    tp = 0
    fp = 0

    # for each MSA, find top-L and compute true/false positives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]

        L = L_mapping[msa_idx]
        kL = min(max(1, int(k*L)), len(y_))
        L_idx = np.argpartition(preds_, -kL)[-kL:]  # [k*L]

        preds_ = np.round(sigmoid(preds_[L_idx]))
        y_ = y_[L_idx]

        if treat_all_preds_positive:
            tp += sum(y_ == 1)
            fp += sum(y_ == 0)
        else:
            tp += sum(np.logical_and(preds_ == 1, y_ == 1))
            fp += sum(np.logical_and(preds_ == 1, y_ == 0))

    top_l_prec = float(tp) / (tp + fp)

    return top_l_prec


def xgb_precision(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray) -> float:
    """
    Custom XGBoost Metric for global precision.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].

    Returns:
        float: Metric value.
    """

    y = dtest.get_label()  # [B]

    msa_indices = np.unique(msa_mapping)
    tp = 0
    fp = 0

    # for each MSA, compute true positives, false negatives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]

        preds_ = np.round(sigmoid(preds_))

        tp += sum(np.logical_and(preds_ == 1, y_ == 1))
        fp += sum(np.logical_and(preds_ == 1, y_ == 0))

    precision = float(tp) / (tp + fp)

    return precision


def xgb_recall(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray) -> float:
    """
    Custom XGBoost Metric for global recall.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].

    Returns:
        float: Metric value.
    """

    y = dtest.get_label()  # [B]

    msa_indices = np.unique(msa_mapping)
    tp = 0
    fn = 0

    # for each MSA, compute true positives, false negatives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]

        preds_ = np.round(sigmoid(preds_))

        tp += sum(np.logical_and(preds_ == 1, y_ == 1))
        fn += sum(np.logical_and(preds_ == 0, y_ == 1))

    recall = float(tp) / (tp + fn)

    return recall


def xgb_F1Score(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray) -> float:
    """
    Custom XGBoost Metric for global F1 score.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtest (xgb.DMatrix): Test data (x: [B, num_maps], y: [B]).
        msa_mapping (np.ndarray): Mapping: Data point -> MSA index [B].

    Returns:
        float: Metric value.
    """

    y = dtest.get_label()  # [B]

    msa_indices = np.unique(msa_mapping)
    tp = 0
    fp = 0
    fn = 0

    # for each MSA, compute true/false positives, false negatives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]

        preds_ = np.round(sigmoid(preds_))

        tp += sum(np.logical_and(preds_ == 1, y_ == 1))
        fp += sum(np.logical_and(preds_ == 1, y_ == 0))
        fn += sum(np.logical_and(preds_ == 0, y_ == 1))

    f1_score = float(tp) / (tp + 0.5 * (fp + fn))

    return f1_score


def xgb_Matthews(preds: np.ndarray, dtest: xgb.DMatrix, msa_mapping: np.ndarray) -> float:
    """
    """

    y = dtest.get_label()
    msa_indices = np.unique(msa_mapping)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # for each MSA, compute true/false positives, false negatives
    for msa_idx in msa_indices:
        mask = msa_mapping == msa_idx  # [B]
        preds_ = preds[mask]
        y_ = y[mask]

        preds_ = np.round(sigmoid(preds_))

        tp += sum(np.logical_and(preds_ == 1, y_ == 1))
        fp += sum(np.logical_and(preds_ == 1, y_ == 0))
        tn += sum(np.logical_and(preds_ == 0, y_ == 0))
        fn += sum(np.logical_and(preds_ == 0, y_ == 1))

    numerator = (float(tp) * float(tn) - float(fp) * float(fn))
    denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0:
        matthews = float('nan')
    else:
        matthews = numerator / denominator
    if matthews < -1.0 or matthews > 1.0:
        print(float(tp))
        print(float(fp))
        print(float(tn))
        print(float(fn))
        raise ValueError("Should be between -1.0 and 1.0")
    return matthews


def xgb_Pearson(preds: np.ndarray, dtest: xgb.DMatrix) -> float:

    y=dtest.get_label()
    
    return pearson_corrcoef(torch.tensor(preds), torch.tensor(y)).item()


def xgb_Spearman(preds: np.ndarray, dtest: xgb.DMatrix) -> float:

    y=dtest.get_label()

    return spearman_corrcoef(torch.tensor(preds), torch.tensor(y)).item()


def xgb_MSE(preds: np.ndarray, dtest: xgb.DMatrix) -> float:

    y=dtest.get_label()

    return mean_squared_error(torch.tensor(preds), torch.tensor(y)).item()


def get_checkpoint_hparams(checkpoint: str, device: Any) -> Dict[str, Any]:
    """
    Loads pytorch-lightning checkpoint and returns hyperparameters.

    Args:
        checkpoint (str): Path to checkpoint.
        device (Any): Device on which model runs.

    Returns:
        Dict[str, Any]: Hyperparameters.
    """

    checkpoint = torch.load(checkpoint, map_location=device)
    h_params = checkpoint['hyper_parameters']

    return h_params


def get_cull_tokens(dataset: datasets.CoCoNetDataset) -> List[str]:
    """
    Returns cull tokens for contact prediction.

    Args:
        dataset (datasets.CoCoNetDataset): CoCoNet dataset that contains the token mapping.

    Returns:
        List[str]: Cull tokens.
    """

    return [dataset.token_mapping[token] for token in ['-', '.', 'START_TOKEN', 'DELIMITER_TOKEN']]


def load_backbone(checkpoint: str, device: Any, dataset: datasets.CoCoNetDataset, cull_tokens: List[str], h_params: Dict[str, Any], downstream_task: str) -> nn.Module:
    """
    Loads pytorch-lightning backbone model.

    Args:
        checkpoint (str): Path to checkpoint.
        device (Any): Device on which model runs.
        dataset (datasets.CoCoNetDataset): CoCoNet dataset that contains the token mapping.
        cull_tokens (List[str]): Cull tokens for contact prediction.
        h_params (Dict[str, Any]): Hyperparameters of the backbone model.
        downstream_task (str): Downstream task (contact, thermostable).

    Returns:
        nn.Module: Loaded backbone model.
    """

    jigsaw_euclid_emb = None
    if 'jigsaw_euclid_emb' in h_params and h_params['jigsaw_euclid_emb']:
        embed_size = checkpoint['state_dict']['task_heads.jigsaw.proj.weight'].size(0)
        jigsaw_euclid_emb = torch.empty((1, embed_size))
    else:
        jigsaw_euclid_emb = None

    if 'jigsaw_disable_delimiter' in h_params:
        jigsaw_delimiter = not h_params['jigsaw_disable_delimiter']
    else:
        jigsaw_delimiter = True

    tasks = []
    if h_params['task_inpainting']:
        tasks.append("inpainting")
    if h_params['task_jigsaw']:
        tasks.append("jigsaw")
    if h_params['task_contrastive']:
        tasks.append("contrastive")
    if h_params['task_jigsaw_boot']:
        tasks.append("jigsaw_boot")

    _, task_heads, task_losses, _, _ = get_tasks(tasks,
                                                 h_params['feature_dim_head'] * h_params['num_heads'],
                                                 subsample_depth=h_params['subsampling_depth'],
                                                 subsample_mode=h_params['subsampling_mode'],
                                                 crop_size=h_params['cropping_size'],
                                                 crop_mode=h_params['cropping_mode'],
                                                 masking=h_params['inpainting_masking_type'],
                                                 p_mask=h_params['inpainting_masking_p'],
                                                 jigsaw_partitions=h_params['jigsaw_partitions'],
                                                 jigsaw_classes=h_params['jigsaw_permutations'],
                                                 jigsaw_linear=not h_params['jigsaw_nonlinear'],
                                                 jigsaw_delimiter=jigsaw_delimiter,
                                                 jigsaw_euclid_emb=jigsaw_euclid_emb,
                                                 simclr_temperature=h_params['contrastive_temperature'],
                                                 jigsaw_boot_ratio=h_params['jigsaw_boot_ratio'],
                                                 per_token=h_params['boot_per_token'],
                                                 boot_same=h_params['boot_same'],
                                                 frozen=h_params['frozen'],
                                                 seq_dist=h_params['seq_dist'],
                                                 )

    num_maps = h_params['num_blocks'] * h_params['num_heads']
    if 'downstream' in checkpoint:
        if downstream_task == 'contact':
            task_heads[downstream_task] = models.self_supervised.msa.modules.ContactHead(num_maps, cull_tokens=cull_tokens)
            task_losses[downstream_task] = None
        elif downstream_task == 'thermostable':
            task_heads[downstream_task] = models.self_supervised.msa.modules.ThermoStableHead(12*64, 1)
            task_losses[downstream_task] = None

    model = models.self_supervised.MSAModel.load_from_checkpoint(
        checkpoint_path=checkpoint,
        num_blocks=h_params['num_blocks'],
        num_heads=h_params['num_heads'],
        feature_dim_head=h_params['feature_dim_head'],
        task_heads=task_heads,
        task_losses=task_losses,
        alphabet_size=len(dataset.token_mapping),
        padding_token=dataset.token_mapping['PADDING_TOKEN'],
        lr=h_params['learning_rate'],
        lr_warmup=h_params['learning_rate_warmup'],
        dropout=0.,
        emb_grad_freq_scale=not h_params['disable_emb_grad_freq_scale'],
        freeze_backbone=True,
        max_seqlen=h_params['cropping_size'],
        h_params=h_params)
    
    if downstream_task == 'contact':
        model.need_attn = True
    elif downstream_task == 'thermostable':
        model.need_attn = False
    
    model.to(device)

    return model


def create_dataloader(mode: str, batch_size: int, subsampling_mode: str, distance_threshold: float, h_params: Dict[str, Any], downstream_task: str, rng_seed: int = 42, disable_train_data_discarding: bool = False, fasta_files: List[str] = [], secondary_window: int = -1) -> DataLoader:
    """
    Creates data loader for downstream task with XGBoost model.

    Args:
        mode (str): Train/Test/Inference.
        batch_size (int): Batch size (currently restricted to 1).
        subsampling_mode (str): Subsampling mode.
        distance_threshold (float): Minimum distance between two atoms in angström that is not considered as a contact.
        h_params (Dict[str, Any]): Hyperparameters of the backbone model.
        downstream_task (str): Downstream task (contact, thermostable).
        rng_seed (int, optional): Seed of the random number generator. Defaults to 42.
        disable_train_data_discarding (bool, optional): Disables the size-based discarding of training data. Defaults to False.
        fasta_files (List[str], optional): List of FASTA files for inference, if chosen as mode. Defaults to [],

    Returns:
        DataLoader: Data loader for downstream task.
    """

    downstream_transform = get_downstream_transforms(downstream_task, subsample_depth=h_params['subsampling_depth'], subsample_mode=subsampling_mode, threshold=distance_threshold, inference=mode == 'inference', secondary_window=secondary_window)
    root = os.environ['DATA_PATH']

    if mode == 'train':
        data_loader_rng = torch.Generator()
        data_loader_rng.manual_seed(rng_seed)
        if downstream_task == 'contact':
            dataset = datasets.CoCoNetDataset(root, mode, transform=downstream_transform, discard_train_size_based=not disable_train_data_discarding,
                                            diversity_maximization=subsampling_mode == 'diversity', max_seq_len=h_params['cropping_size'],
                                            min_num_seq=h_params['subsampling_depth'],
                                            secondary_window=secondary_window)
        elif downstream_task == 'thermostable':
            dataset=datasets.challData_lab(root,
                    mode,
                    transform=downstream_transform,
                    discard_train_size_based=not disable_train_data_discarding,
                    diversity_maximization=subsampling_mode == 'diversity',
                    max_seq_len=h_params['cropping_size'],
                    min_num_seq=h_params['subsampling_depth'],
                    secondary_window=secondary_window)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          worker_init_fn=partial(data_loader_worker_init, rng_seed=rng_seed),
                          generator=data_loader_rng,
                          pin_memory=False)

    elif mode == 'test':
        if downstream_task == 'contact':
            dataset = datasets.CoCoNetDataset(root, mode, transform=downstream_transform, diversity_maximization=subsampling_mode == 'diversity', secondary_window=secondary_window)
        elif downstream_task == 'thermostable':
            dataset=datasets.challData_lab(root,
                    mode,
                    transform=downstream_transform,
                    discard_train_size_based=not disable_train_data_discarding,
                    diversity_maximization=subsampling_mode == 'diversity',
                    max_seq_len=h_params['cropping_size'],
                    min_num_seq=h_params['subsampling_depth'],
                    secondary_window=secondary_window)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False)
    elif mode == 'inference':
        assert downstream_task == 'contact'
        dataset = datasets.InferenceDataset(fasta_files, transform=downstream_transform)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False)
    else:
        raise ValueError("Unknown dataloader mode: %s" % mode)


def compute_attn_maps(model: nn.Module, dataloader: DataLoader, cull_tokens: List[str], diag_shift: int, h_params: Dict[str, Any], device: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes attention maps for all data items.

    Args:
        model (nn.Module): Backbone model.
        dataloader (DataLoader): Data loader for downstream task.
        cull_tokens (List[str]): Cull tokens for contact prediction.
        diag_shift (int): Diagonal offset by which contacts are ignored.
        h_params (Dict[str, Any]): Hyperparameters of the backbone model.
        device (Any): Device on which model runs.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Attention maps [B*L*L/2, num_maps]; targets [B*L*L/2]; msa_mapping [B*L*L]; msa_mask [B*L*L]; msa_mapping_filtered [B*L*L/2], L_mapping [B].
    """

    attn_maps_list = []
    targets_list = []
    msa_mapping_list = []
    msa_mask_list = []
    msa_mapping_filtered_list = []
    L_mapping_list = []

    num_maps = h_params['num_blocks'] * h_params['num_heads']

    for idx, (x, y) in enumerate(dataloader):
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        with torch.no_grad():
            _, attn_maps = model(x['msa'], x.get('padding_mask', None), x.get('aux_features', None))

        B, _, L = x['msa'].shape
        assert B == 1

        mask = torch.ones((L, ), device=device)
        for token in cull_tokens:
            mask -= (x['msa'][:, 0].reshape((L, )) == token).int()
        mask = mask.bool()
        degapped_L = int(mask.sum())
        mask = torch.reshape(mask, (B, 1, L))
        mask = torch.logical_and(mask.reshape((B, 1, L)).expand(B, L, L), mask.reshape((B, L, 1)).expand(B, L, L))
        mask = mask.reshape((B, 1, L, L)).expand((B, num_maps, L, L))

        attn_maps = torch.cat([m.squeeze(dim=2) for m in attn_maps], dim=1)  # [1, num_maps, L, L]
        attn_maps = attn_maps.masked_select(mask).reshape(B, num_maps, degapped_L, degapped_L)
        attn_maps = torch.permute(attn_maps, (0, 2, 3, 1))  # [1, L, L, num_maps]

        assert num_maps == attn_maps.shape[-1]

        attn_maps_triu = attn_maps.view(-1, num_maps)  # [1*L*L, num_maps]
        attn_maps_tril = torch.permute(attn_maps, (0, 2, 1, 3)).reshape(-1, num_maps)  # [1*L*L, num_maps]
        if 'contact' not in y:
            target = None
        else:
            target = y['contact'].view(-1)  # [1*L*L]

        msa_mapping = torch.full((B * degapped_L * degapped_L, ), idx, dtype=torch.int64, device=attn_maps.device)  # [1*L*L]

        # exclude unknown target points, apply diag shift, averge over both triangle matrices
        if 'contact' not in y:
            mask = torch.ones((B, degapped_L, degapped_L), dtype=torch.bool, device=attn_maps.device)  # [1, L, L]
        else:
            mask = y['contact'] != -1
        mask_triu = torch.triu(torch.ones_like(mask), diag_shift+1).view(-1)  # [1*L*L]

        mask = mask.view(-1)  # [1*L*L]
        mask_attn_maps = mask[mask_triu]
        mask_target = torch.logical_and(mask, mask_triu)

        attn_maps_triu = attn_maps_triu[mask_triu, :]
        attn_maps_tril = attn_maps_tril[mask_triu, :]

        attn_maps = 0.5 * (attn_maps_triu + attn_maps_tril)
        attn_maps = attn_maps[mask_attn_maps, :]
        if target is not None:
            target = target[mask_target]
        msa_mapping_filtered = msa_mapping[mask_target]

        attn_maps_list.append(attn_maps)
        targets_list.append(target)
        msa_mapping_list.append(msa_mapping)
        msa_mask_list.append(mask_target)
        msa_mapping_filtered_list.append(msa_mapping_filtered)
        L_mapping_list.append(degapped_L)

    attn_maps = torch.cat(attn_maps_list)  # [B*L*L/2, num_maps]
    if targets_list[0] is not None:
        targets = torch.cat(targets_list)  # [B*L*L/2]
    else:
        targets = None
    msa_mapping = torch.cat(msa_mapping_list)  # [B*L*L]
    msa_mask = torch.cat(msa_mask_list)  # [B*L*L]
    msa_mapping_filtered = torch.cat(msa_mapping_filtered_list)  # [B*L*L/2]

    attn_maps = attn_maps.cpu().numpy()

    if targets is not None:
        targets = targets.cpu().numpy()
    msa_mapping = msa_mapping.cpu().numpy()
    msa_mask = msa_mask.cpu().numpy()
    msa_mapping_filtered = msa_mapping_filtered.cpu().numpy()
    L_mapping = np.array(L_mapping_list)

    return attn_maps, targets, msa_mapping, msa_mask, msa_mapping_filtered, L_mapping


def compute_latent(model: nn.Module, dataloader: DataLoader, device: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes latent representations for all data items.

    Args:
        model (nn.Module): Backbone model.
        dataloader (DataLoader): Data loader for downstream task.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Latent, targets.
    """

    latent_list = []
    targets_list = []

    for idx, (x, y) in enumerate(dataloader):
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)
            
        assert 'thermostable' in y
        # add column of -1 to mask out start token
        B, E, _ = y['thermostable'].shape
        y_extended = torch.cat((torch.full((B, E, 1), -0.0025, dtype=y['thermostable'].dtype), y['thermostable']), dim=2)

        with torch.no_grad():
            latent = model(x['msa'], x.get('padding_mask', None), x.get('aux_features', None))
            
        mask = y_extended != -0.0025
        y['thermostable'] = y_extended[mask]
        latent=latent[mask,:]

        B, E, L = x['msa'].shape
        assert B == 1

        target=y['thermostable']
        
        latent_list.append(latent)
        targets_list.append(target)

    latent=torch.cat(latent_list)
    if targets_list[0] is not None:
        targets = torch.cat(targets_list)  # [B*L*L/2]
    else:
        targets = None

    latent=latent.cpu().numpy()
    if targets is not None:
        targets = targets.cpu().numpy()
    
    return latent, targets


def metric_wrapper_contact(preds: np.ndarray, dtrain: xgb.DMatrix, metric: str, msa_mappings: Tuple[np.ndarray, np.ndarray],
        L_mapping: np.ndarray, 
        k: float = 1., treat_all_preds_positive: bool = False) -> Tuple[str, float]:
    """
    Custom XGBoost Metric for contact downstream task.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtrain (xgb.DMatrix): Training data (x: [B, num_maps], y: [B]).
        metric (str): Metric.
        msa_mappings (Tuple[np.ndarray, np.ndarray]): Mapping: Data point -> MSA index [B] (Train, Val).
        L_mapping (np.ndarray): Mapping: MSA index -> MSA L.
        k (float, optional): Coefficient k that is used in computing the top-(k*L)-precision. Defaults to 1.
        treat_all_preds_positive (bool, optional): Whether all non-ignored preds are treated as positives, analogous to the CocoNet paper. Defaults to False.

    Returns:
        Tuple[str, float]: Metric name; metric value.
    """
    assert metric is not None

    y = dtrain.get_label()  # [B]

    # Dirty hack: Find out by data length, whether training or validation is active. Only works, if training and validation dataset have different lengths.
    B = len(y)
    if len(msa_mappings[0]) == B:
        msa_mapping = msa_mappings[0]
    elif len(msa_mappings[1]) == B:
        msa_mapping = msa_mappings[1]
    else:
        raise ValueError("Given data length does not match to msa_mappings: %d != (%d, %d)" % (B, len(msa_mappings[0]), len(msa_mappings[1])))

    metrics = {'toplprec': xgb_topkLPrec, 'f1': xgb_F1Score, 'matthews': xgb_Matthews}
    
    if metric == 'toplprec':
        value = metrics[metric](preds, dtrain, msa_mapping, L_mapping, k=k, treat_all_preds_positive=treat_all_preds_positive)
        description = 'top-%sL-Prec' % str(k)
    else:
        value = metrics[metric](preds, dtrain)
        description = metric

    return description, value


def metric_wrapper_thermo(preds: np.ndarray, dtrain: xgb.DMatrix, metric: str) -> Tuple[str, float]:
    """
    Custom XGBoost Metric for thermostable downstream task.

    Args:
        preds (np.ndarray): Predictions [B] as logits.
        dtrain (xgb.DMatrix): Training data (x: [B, num_maps], y: [B]).
        metric (str): Metric.

    Returns:
        Tuple[str, float]: Metric name; metric value.
    """
    assert metric is not None

    y = dtrain.get_label()  # [B]

    metrics = {'mse': xgb_MSE, 'pcorr': xgb_Pearson, 'scorr': xgb_Spearman}

    value = metrics[metric](preds, dtrain)
    description = metric

    return description, value