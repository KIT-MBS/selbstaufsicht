import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Selbstaufsicht Weakly Supervised Contact Prediction Script')
    # Pre-trained model
    parser.add_argument('--log-path', type=str, help="Path to logging data of the specific downstream run to be analyzed.")

    args = parser.parse_args()

    fold_dirs = [item for item in os.listdir(args.log_path)]
    num_folds = len(fold_dirs)

    num_epochs_arr = []
    top_l_prec_arr = []


    for fold_dir in fold_dirs:
        checkpoint_path = os.path.join(args.log_path, fold_dir, 'checkpoints')
        checkpoint = [item for item in os.listdir(checkpoint_path) if 'downstream' in item]
        assert len(checkpoint) == 1
        checkpoint = checkpoint[0].replace('downstream-epoch=', '').replace('-contact_validation_topLprec', '').replace('.ckpt', '')
        splits = checkpoint.split('=')
        assert len(splits) == 2
        num_epochs, top_l_prec = splits[0], splits[1]
        num_epochs = float(num_epochs)
        top_l_prec = float(top_l_prec)
        num_epochs_arr.append(num_epochs)
        top_l_prec_arr.append(top_l_prec)

    num_epochs_arr = np.array(num_epochs_arr)
    top_l_prec_arr = np.array(top_l_prec_arr)

    num_epochs_mean = num_epochs_arr.mean()
    num_epochs_std = num_epochs_arr.std(ddof=1)

    top_l_prec_mean = top_l_prec_arr.mean()
    top_l_prec_std = top_l_prec_arr.std(ddof=1)

    print(f"{num_folds}-fold cross-validation: {top_l_prec_mean} +- {top_l_prec_std} top-L-precision after {num_epochs_mean} +- {num_epochs_std} epochs.")

if __name__ == '__main__':
    main()