import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    # Pre-trained model
    parser.add_argument('--log-path', type=str, help="Path to logging data of the specific xgboost downstream run to be evaluated.")
    parser.add_argument('--task', default='contact', type=str, help="Downstream task ('contact', 'thermostable')")

    args = parser.parse_args()

    fold_files = [item for item in os.listdir(args.log_path) if item.startswith('cv_')]
    num_folds = len(fold_files)

    num_trees_arr = []
    metric_arr = []

    for fold_file in fold_files:
        fold_path = os.path.join(args.log_path, fold_file)
        df = pd.read_csv(fold_path)
        val_col_name = list(df.columns.values)[-1]
        idx = df[val_col_name].idxmax()
        metric = float(df[val_col_name].iloc[[idx]])

        num_trees = float(idx+1)
        metric = float(metric)
        num_trees_arr.append(num_trees)
        metric_arr.append(metric)

    num_trees_arr = np.array(num_trees_arr)
    metric_arr = np.array(metric_arr)

    num_trees_mean = num_trees_arr.mean()
    num_trees_std = num_trees_arr.std(ddof=1)

    metric_mean = metric_arr.mean()
    metric_std = metric_arr.std(ddof=1)

    if args.task == 'contact':
        print(f"{num_folds}-fold cross-validation: {metric_mean} +- {metric_std} top-L-precision with {num_trees_mean} +- {num_trees_std} trees.")
    elif args.task == 'thermostable':
        print(f"{num_folds}-fold cross-validation: {metric_mean} +- {metric_std} Pearson-corrcoef with {num_trees_mean} +- {num_trees_std} trees.")

if __name__ == '__main__':
    main()
