import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    # Pre-trained model
    parser.add_argument('--log-path', type=str, help="Path to logging data of the specific xgboost downstream run to be evaluated.")

    args = parser.parse_args()

    fold_files = [item for item in os.listdir(args.log_path) if item.startswith('cv_')]
    num_folds = len(fold_files)

    num_trees_arr = []
    top_l_prec_arr = []

    for fold_file in fold_files:
        fold_path = os.path.join(args.log_path, fold_file)
        df = pd.read_csv(fold_path)
        val_col_name = list(df.columns.values)[-1]
        idx = df[val_col_name].idxmax()
        top_l_prec = float(df[val_col_name].iloc[[idx]])

        num_trees = float(idx+1)
        top_l_prec = float(top_l_prec)
        num_trees_arr.append(num_trees)
        top_l_prec_arr.append(top_l_prec)

    num_trees_arr = np.array(num_trees_arr)
    top_l_prec_arr = np.array(top_l_prec_arr)

    num_trees_mean = num_trees_arr.mean()
    num_trees_std = num_trees_arr.std(ddof=1)

    top_l_prec_mean = top_l_prec_arr.mean()
    top_l_prec_std = top_l_prec_arr.std(ddof=1)

    print(f"{num_folds}-fold cross-validation: {top_l_prec_mean} +- {top_l_prec_std} top-L-precision with {num_trees_mean} +- {num_trees_std} trees.")


if __name__ == '__main__':
    main()
