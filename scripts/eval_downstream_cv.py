import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    # Pre-trained model
    parser.add_argument('--log-path', type=str, help="Path to logging data of the specific downstream run to be evaluated.")
    parser.add_argument('--task', default='contact', type=str, help="Downstream task ('contact', 'thermostable')")
    parser.add_argument('--monitor-metric', default="", type=str, help='Metric used for early stopping on validation data.')

    args = parser.parse_args()

    fold_dirs = [item for item in os.listdir(args.log_path)]
    num_folds = len(fold_dirs)

    num_epochs_arr = []
    metric_arr = []

    for fold_dir in fold_dirs:
        checkpoint_path = os.path.join(args.log_path, fold_dir, 'checkpoints')
        checkpoint = [item for item in os.listdir(checkpoint_path) if 'downstream' in item and f'{args.monitor_metric}=' in item]
        assert len(checkpoint) == 1
        if args.task == 'contact':
            checkpoint = checkpoint[0].replace('downstream-epoch=', '').replace(f'-contact_validation_{args.monitor_metric}', '').replace('.ckpt', '')
        elif args.task == 'thermostable':
            checkpoint = checkpoint[0].replace('downstream-epoch=', '').replace(f'-{args.monitor_metric}', '').replace('.ckpt', '')
        splits = checkpoint.split('=')
        assert len(splits) == 2
        num_epochs, metric = splits[0], splits[1]
        num_epochs = float(num_epochs)
        metric = float(metric)
        num_epochs_arr.append(num_epochs)
        metric_arr.append(metric)

    num_epochs_arr = np.array(num_epochs_arr)
    metric_arr = np.array(metric_arr)

    num_epochs_mean = num_epochs_arr.mean()
    num_epochs_std = num_epochs_arr.std(ddof=1)

    metric_mean = metric_arr.mean()
    metric_std = metric_arr.std(ddof=1)

    if args.task == 'contact':
        print(f"{num_folds}-fold cross-validation: {metric_mean} +- {metric_std} top-L-precision after {num_epochs_mean} +- {num_epochs_std} epochs.")
    elif args.task == 'thermostable':
        print(f"{num_folds}-fold cross-validation: {metric_mean} +- {metric_std} Pearson-corrcoef after {num_epochs_mean} +- {num_epochs_std} epochs.")


if __name__ == '__main__':
    main()
