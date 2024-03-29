import math
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import sklearn.metrics as skl_metrics
import torch
from torch import nn

from selbstaufsicht.modules import Transmorpher2d, TransmorpherBlock2d


class MSAModel(pl.LightningModule):
    """
    Model for pre-training on multiple sequence alignments of biological sequences
    """

    def __init__(
            self,
            num_blocks: int = 12,
            num_heads: int = 12,
            dim_head: int = 64,
            attention: str = 'tied',
            activation: str = 'relu',
            layer_norm_eps: float = 1e-5,
            alphabet_size: int = 7,
            lr: float = 1e-4,
            lr_warmup: int = 16000,
            dropout: float = 0.1,
            padding_token: int = None,
            emb_grad_freq_scale: bool = False,
            pos_padding_token: int = 0,
            max_seqlen: int = 5000,
            h_params: Dict[str, Any] = None,
            task_heads: Dict[str, nn.Module] = None,
            task_losses: Dict[str, nn.Module] = None,
            task_loss_weights: Dict[str, float] = None,
            train_metrics: Dict[str, nn.ModuleDict] = None,
            val_metrics: Dict[str, nn.ModuleDict] = None,
            need_attn: bool = False,
            freeze_backbone: bool = False,
            device: Union[str, torch.device] = None,
            dtype: torch.dtype = None) -> None:
        """
        Initializes backbone model for pre-training on multiple sequence alignments of biological sequences.

        Args:
            num_blocks (int, optional): Number of consecutive Transmorpher blocks. Defaults to 12.
            num_heads (int, optional): Number of parallel Transmorpher heads. Defaults to 12.
            dim_head (int, optional): Embedding dimensionality of a single Transmorpher head. Defaults to 64.
            attention (str, optional): Used attention mechanism. Defaults to 'tied'.
            activation (str, optional): Used activation function. Defaults to 'relu'.
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            alphabet_size (int, optional): Input alphabet size. Defaults to 7.
            lr (float, optional): Initial learning rate. Defaults to 1e-4.
            lr_warmup (int, optional): Warmup parameter for inverse square root rule of learning rate scheduling. Defaults to 16000.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            padding_token (int, optional): Numerical token that is used for padding in evolutionary and sequence dimensions. Defaults to None.
            emb_grad_freq_scale (bool, optional): flag whether to scale gradients by the inverse of frequency of the tokens in the mini-batch
            pos_padding_token (int, optional): Numerical token that is used for padding in positional embedding in auxiliary input
            max_seqlen (int, optional): maximum sequence length for learned positional embedding
            h_params (Dict[str, Any], optional): Hyperparameters for logging. Defaults to None.
            task_heads (nn.ModuleDict[str, nn.Module], optional): Head modules for upstream tasks. Defaults to None.
            task_losses (Dict[str, nn.Module], optional): Loss functions for upstream tasks. Defaults to None.
            task_loss_weights (Dict[str, float], optional): per task loss weights. Defaults to None.
            train_metrics (Dict[str, nn.ModuleDict], optional): Training metrics for upstream tasks. Defaults to None.
            val_metrics (Dict[str, nn.ModuleDict], optional): Validation metrics for upstream tasks. Defaults to None.
            need_attn (bool, optional): Whether to extract attention maps or not. Defaults to False.
            freeze_backbone (bool, optional): Freezes backbone parameters during downstream task. Defaults to False.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.

        Raises:
            NotImplementedError: If need_attn=True: Extracting attention maps not yet implemented.
        """

        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        d = num_heads * dim_head

        self.embedding = nn.Embedding(alphabet_size, d, padding_idx=padding_token, scale_grad_by_freq=emb_grad_freq_scale)
        self.positional_embedding = nn.Embedding(max_seqlen, d, padding_idx=pos_padding_token)
        block = TransmorpherBlock2d(dim_head, num_heads, 2 * dim_head * num_heads, dropout=dropout, attention=attention, activation=activation, layer_norm_eps=layer_norm_eps, **factory_kwargs)
        self.backbone = Transmorpher2d(block, num_blocks, nn.LayerNorm(d, eps=layer_norm_eps, **factory_kwargs))
        self.tasks = None
        # TODO adapt to non-simultaneous multi task training (all the heads will be present in model, but not all targets in one input)
        if task_heads is not None:
            self.tasks = [t for t in task_heads.keys()]

        self.task_loss_weights = task_loss_weights
        if self.tasks is not None and self.task_loss_weights is None:
            self.task_loss_weights = {t: 1. for t in self.tasks}
        if self.task_loss_weights is not None:
            self.task_loss_weights = {t: self.task_loss_weights[t] / (sum(self.task_loss_weights.values())) for t in self.task_loss_weights}

        self.task_heads = task_heads
        self.losses = task_losses
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = None
        self.downstream_loss_device_flag = False
        if task_heads is not None:
            assert self.task_heads.keys() == self.losses.keys()
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.need_attn = need_attn
        self.freeze_backbone = freeze_backbone
        self.save_hyperparameters(h_params)
        self.log_images = False
        self.max_seqlen=max_seqlen

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None, aux_features: torch.Tensor = None) -> torch.Tensor:
        """
        Receives cropped, subsampled and tokenized MSAs as input data, passes them through several layers with attention mechanism to yield a latent representation.

        Args:
            x (torch.Tensor): Cropped, subsampled and tokenized input MSAs [B, E, L].
            padding_mask (torch.Tensor, optional): Bool tensor that points out locations of padded values in the input data [B, E, L]. Defaults to None.
            aux_features (torch.Tensor, optional): Auxiliary features (positional encoding). Defaults to None.

        Returns:
            torch.Tensor: Latent representation [B, E, L, D].
        """

        # NOTE feature dim = -1

        x = self.embedding(x) + self.positional_embedding(aux_features)
        return self.backbone(x, padding_mask, self.need_attn)

    def _step(self, batch_data: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], batch_idx: int, test: bool = False) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        Performs a single training or validation step: First passes cropped, subsampled and tokenized MSAs through the backbone model,
        whose latent representation output is then passed through the upstream task related head models.
        Eventually, using task specific loss function and further metrics, the obtained prediction results are evaluated against the corresponding label data.

        Args:
            batch_data (Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]): Input data, Label data.
            batch_idx (int): Batch number.
            test (bool, optional): Whether testing is enabled. Defaults to False.

        Returns:
            Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]: Input, target, predictions, summed loss across all upstream tasks
        """

        x, y = batch_data
        assert not (self.training and test)

        if self.training:
            mode = "training"
            metrics = self.train_metrics
        else:
            mode = "validation"
            metrics = self.val_metrics
        if test:
            assert self.test_metrics is not None
            mode = "test"
            metrics = self.test_metrics

        latent = None
        if 'contact' in self.tasks or 'thermostable' in self.tasks:
            if not self.downstream_loss_device_flag and not 'thermostable' in self.tasks and hasattr(self.losses['contact'], 'weight'):
                self.losses['contact'].weight = self.losses['contact'].weight.to(self.device)
                self.downstream_loss_device_flag = True

            # NOTE (un)frozen weights for all modules except contact head
            with torch.set_grad_enabled(not self.freeze_backbone and self.training):
                if 'contact' in self.tasks:
                    latent, attn_maps = self(x['msa'], x.get('padding_mask', None), x.get('aux_features', None))
                    x['attn_maps'] = attn_maps
                elif 'thermostable' in self.tasks:
                    # add column of -1 to mask out start token
                    B, E, _ = y['thermostable'].shape
                    y_extended = torch.cat((torch.full((B, E, 1), -0.0025, dtype=y['thermostable'].dtype).to(self.device), y['thermostable']), dim=2)
                    mask = y_extended != -0.0025
                    latent = self(x['msa'], x.get('padding_mask', None), x.get('aux_features', None))
                    y['thermostable'] = y_extended[mask]
                    latent=latent[mask,:]
        else:
            latent = self(x['msa'], x.get('padding_mask', None), x.get('aux_features', None))

        if 'contrastive' in self.tasks:
            y['contrastive'] = None
        
        preds = {task: self.task_heads[task](latent, x) for task in self.tasks}
        lossvals = {task: self.losses[task](preds[task], y[task]) for task in self.tasks}
        for task in self.tasks:
            for m in metrics[task]:
                # NOTE: ContactHead output is symmetrized raw scores, so Sigmoid has to be applied explicitly
                if task == 'contact':
                    metrics[task][m](torch.sigmoid(preds[task]), y[task])
                elif task == 'thermostable':
                    metrics[task][m](torch.flatten(preds[task]), torch.flatten(y[task]))
                else:
                    metrics[task][m](preds[task], y[task])
                if 'confmat' not in m and 'unreduced' not in m:
                    self.log(f'{task}_{mode}_{m}', metrics[task][m], on_step=self.training, on_epoch=True)
        loss = sum([self.task_loss_weights[task] * lossvals[task] for task in self.tasks])
        for task in self.tasks:
            self.log(f'{task}_{mode}_loss', lossvals[task], on_step=self.training, on_epoch=True)

        self.log(f'{mode}_loss', loss, on_step=self.training, on_epoch=True)
        if 'contact' in self.tasks:
            # NOTE invoke tensorboard with --samples_per_plugin images=<num_batches> to show more images
            if mode == 'validation':
                # NOTE plot attention maps per block summed over heads
                if self.log_images:
                    for i, a in enumerate(x['attn_maps']):
                        plt.figure()
                        fig = sns.heatmap(torch.sum(torch.squeeze(a[0], dim=1), dim=0).cpu().numpy(), fmt='').get_figure()
                        plt.close(fig)
                        self.logger.experiment.add_figure(f'map_block_{i}', fig, self.current_epoch)

                    # NOTE plot contact prediction scores
                    plt.figure()
                    fig = sns.heatmap(preds['contact'][0, 1].cpu().numpy(), fmt='').get_figure()
                    plt.close(fig)
                    self.logger.experiment.add_figure('contact_pred', fig, self.current_epoch)
                    # NOTE plot contact predictions
                    plt.figure()
                    fig = sns.heatmap((preds['contact'][0, 1] >= 0.5).float().cpu().numpy(), fmt='').get_figure()
                    plt.close(fig)
                    self.logger.experiment.add_figure('contact_pred', fig, self.current_epoch)

                    # NOTE plot top L contact predictions
                    L = y['contact'].size(1)
                    preds_ = preds['contact'][0, 1]
                    preds_ = torch.triu(preds_, 4) + torch.tril(torch.full_like(preds_, -torch.inf), 4)
                    preds_ = preds_.view(L*L)
                    topl_preds_ = torch.zeros_like(preds_)
                    val, idx = torch.topk(preds_, L, dim=-1)  # [L]
                    topl_preds_[idx] = 1.
                    topl_preds_ = topl_preds_.view(L, L)

                    plt.figure()
                    fig = sns.heatmap(topl_preds_.cpu().numpy(), fmt='').get_figure()
                    plt.close(fig)
                    self.logger.experiment.add_figure('topl_contacts', fig, self.current_epoch)

                    # NOTE plot contact prediction targets
                    plt.figure()
                    fig = sns.heatmap(y['contact'][0].cpu().numpy(), fmt='').get_figure()
                    plt.close(fig)
                    self.logger.experiment.add_figure('contact_target', fig, self.current_epoch)

        for task in self.tasks:
            preds[task] = preds[task].detach()
        if 'contact' in self.tasks and not self.freeze_backbone:
            x['attn_maps'] = [row_map.detach() for row_map in x['attn_maps']]
        out = {'input': x, 'preds': preds, 'target': y, 'loss': loss, 'test': test}

        return out

    def _create_confusion_matrix(self, conf_mat_metric: Any, mode: str) -> None:
        """
        Computes and plots confusion matrix.

        Args:
            conf_mat_metric (Any): Confusion matrix metric.
            mode (str): Training or validation or test.
        """

        conf_mat = conf_mat_metric.compute()
        num_total = conf_mat.sum()
        annotations = np.array([
                                [
                                    "TP\n%.2E\n(%.2f%%)" % (conf_mat[0, 0], 100. * conf_mat[0, 0] / num_total),
                                    "FN\n%.2E\n(%.2f%%)" % (conf_mat[0, 1], 100. * conf_mat[0, 1] / num_total)
                                ],
                                [
                                    "FP\n%.2E\n(%.2f%%)" % (conf_mat[1, 0], 100. * conf_mat[1, 0] / num_total),
                                    "TN\n%.2E\n(%.2f%%)" % (conf_mat[1, 1], 100. * conf_mat[1, 1] / num_total)]
                                ]
                               )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(conf_mat.numpy(), annot=annotations, cmap='coolwarm', fmt='').get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure("contact_%s_confmat" % mode, fig_, self.current_epoch)
        conf_mat_metric.reset()

    def _create_roc_curve(self, preds: List[torch.Tensor], targets: List[torch.Tensor], mode: str) -> None:
        """
        Computes and plots ROC curve.

        Args:
            preds (List[torch.Tensor]): Predictions [B, 2, L, L].
            target (List[torch.Tensor]): Targets [B, L, L].
            mode (str): Training or validation or test.
        """

        preds_ = torch.cat([torch.exp(tmp[:, 1, :, :]).flatten() for tmp in preds])
        target_ = torch.cat([tmp.flatten() for tmp in targets])

        preds_ = preds_[target_ != -1]
        preds_[preds_ == -torch.inf] = torch.finfo(preds_.dtype).min
        target_ = target_[target_ != -1]

        preds_ = preds_.cpu().numpy()
        target_ = target_.cpu().numpy()

        fpr, tpr, threshold = skl_metrics.roc_curve(target_, preds_)
        auc = skl_metrics.auc(fpr, tpr)

        fig_ = plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.close(fig_)

        self.logger.experiment.add_figure("contact_%s_roc" % mode, fig_, self.current_epoch)

    def _create_correctness_histogram(self, conf_mat_metric: Any, mode: str) -> None:
        """
        Computes and plots correctness histogram, i.e., TP, FP, TN, FN per MSA.

        Args:
            conf_mat_metric (Any): Confusion matrix metric (unreduced).
            mode (str): Training or validation or test.
        """

        conf_mat = conf_mat_metric.compute()
        num_msa = conf_mat.shape[2]
        num_total_per_msa = conf_mat.sum(dim=(0, 1))
        conf_mat = conf_mat / num_total_per_msa
        conf_mat = conf_mat.numpy()
        x_range = np.arange(num_msa)

        df = pd.DataFrame(np.c_[conf_mat[0, 0], conf_mat[1, 0], conf_mat[1, 1], conf_mat[0, 1]], index=x_range, columns=["TP", "FP", "TN", "FN"])
        ax = df.plot.bar(figsize=(num_msa*0.8, 7), rot=0, color=['red', 'royalblue', 'lime', 'fuchsia'])
        fig = ax.get_figure()
        plt.legend(loc='upper right')

        plt.close(fig)

        self.logger.experiment.add_figure("contact_%s_correctness_histogram" % mode, fig, self.current_epoch)
        conf_mat_metric.reset()

    def _create_topLprec_histogram(self, top_l_precision_metric: Any, mode: str) -> None:
        """
        Computes and plots top-L precision histogram, i.e., top-L precision per MSA.

        Args:
            top_l_precision_metric (Any): Top-L precision metric (unreduced).
            mode (str): Training or validation or test.
        """

        top_l_precision = top_l_precision_metric.compute()
        num_msa = top_l_precision.shape[0]
        top_l_precision = top_l_precision.cpu().numpy()
        top_l_precision_mean = top_l_precision.mean()
        x_range = np.arange(num_msa)

        fig_ = plt.figure(figsize=(num_msa, 7))
        plt.bar(x_range, top_l_precision, color='r', label="Top-L precision")
        plt.plot([min(x_range), max(x_range)], [top_l_precision_mean, top_l_precision_mean], color='b', linewidth=3., label="Mean")
        plt.legend(loc='upper right')
        plt.close(fig_)

        self.logger.experiment.add_figure("contact_%s_topLPrec_histogram" % mode, fig_, self.current_epoch)
        top_l_precision_metric.reset()

    def _epoch_end(self, outputs: List[Any]) -> None:
        """
        Is invoked at the end of an epoch. Computes metrics for imbalanced datasets, if \"contact\" is in tasks.

        Args:
            outputs (List[Any]): Outputs from training/validation steps.
        """

        if 'contact' in self.tasks:
            if self.training:
                mode = "training"
                metrics = self.train_metrics
            else:
                mode = "validation"
                metrics = self.val_metrics
            if len(outputs) > 0 and outputs[0]['test']:
                mode = "test"
                metrics = self.test_metrics

            self._create_confusion_matrix(metrics['contact']['confmat'], mode)
            self._create_correctness_histogram(metrics['contact']['confmat_unreduced'], mode)
            self._create_topLprec_histogram(metrics['contact']['topLprec_unreduced'], mode)
            if len(outputs) > 0:
                self._create_roc_curve([tmp['preds']['contact'] for tmp in outputs], [tmp['target']['contact'] for tmp in outputs], mode)

    def training_step(self, batch_data: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step: First passes cropped, subsampled and tokenized MSAs through the backbone model,
        whose latent representation output is then passed through the upstream task related head models.
        Eventually, using task specific loss function and further metrics, the obtained prediction results are evaluated against the corresponding label data.

        Args:
            batch_data (Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]): Input data, Label data.
            batch_idx (int): Batch number.

        Returns:
            torch.Tensor: Summed loss across all upstream tasks
        """

        return self._step(batch_data, batch_idx)

    def validation_step(self, batch_data: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], batch_idx: int) -> None:
        """
        Performs a single validation step: First passes cropped, subsampled and tokenized MSAs through the backbone model,
        whose latent representation output is then passed through the upstream task related head models.
        Eventually, using task specific loss function and further metrics, the obtained prediction results are evaluated against the corresponding label data.

        Args:
            batch_data (Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]): Input data, Label data.
            batch_idx (int): Batch number.
        """

        return self._step(batch_data, batch_idx)

    def test_step(self, batch_data: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], batch_idx: int) -> None:
        """
        Performs a single test step: First passes cropped, subsampled and tokenized MSAs through the backbone model,
        whose latent representation output is then passed through the upstream task related head models.
        Eventually, using task specific loss function and further metrics, the obtained prediction results are evaluated against the corresponding label data.

        Args:
            batch_data (Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]): Input data, Label data.
            batch_idx (int): Batch number.
        """

        return self._step(batch_data, batch_idx, test=True)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        Is invoked at the end of a training epoch.

        Args:
            outputs (List[Any]): Outputs from training steps.
        """

        self._epoch_end(outputs)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Is invoked at the end of a validation epoch.

        Args:
            outputs (List[Any]): Outputs from validation steps.
        """

        self._epoch_end(outputs)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """
        Is invoked at the end of a test epoch.

        Args:
            outputs (List[Any]): Outputs from test steps.
        """

        self._epoch_end(outputs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures optimization algorithm.

        Returns:
            Dict[str, Any]: Optimization algorithm, lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        class inverse_square_root_rule():
            def __init__(self, warmup: int) -> None:
                """
                Initializes InverseSquareRootRule used for lr scheduling.

                Args:
                    warmup (int): Warmup parameter.
                """

                self.warmup = warmup

            def __call__(self, i: int) -> float:
                """
                Performs InverseSquareRootRule used for lr scheduling.

                Args:
                    i (int): Epoch number.

                Returns:
                    float: Multiplicative factor for lr scheduling.
                """

                if self.warmup > 0:
                    return min((i + 1) / self.warmup, math.sqrt(self.warmup / (i + 1)))
                else:
                    return 1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, inverse_square_root_rule(self.lr_warmup))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
