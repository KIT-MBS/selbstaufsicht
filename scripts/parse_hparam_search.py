import argparse
from enum import Enum
import glob
import os
import pandas as pd
import pickle
from tensorflow.core.util import event_pb2
from tensorflow.data import TFRecordDataset
import torch

# Parsing script for random-search-based hyperparameter optimization.

# DEFINE PARSER
parser = argparse.ArgumentParser(description='Selbstaufsicht Random-Search Hyperparameter Optimization - Parsing Script')
parser.add_argument('--acc-threshold-inpainting-train', default=0.8, type=float, help="Minimum inpainting training accuracy for acceptance")
parser.add_argument('--acc-threshold-inpainting-val', default=0.8, type=float, help="Minimum inpainting validation accuracy for acceptance")
parser.add_argument('--acc-threshold-jigsaw-train', default=0.95, type=float, help="Minimum jigsaw training accuracy for acceptance")
parser.add_argument('--acc-threshold-jigsaw-val', default=0.95, type=float, help="Minimum jigsaw validation accuracy for acceptance")
parser.add_argument('--task-inpainting', action='store_true', help="Activates the inpainting task")
parser.add_argument('--task-jigsaw', action='store_true', help="Activates the jigsaw task")
parser.add_argument('--task-contrastive', action='store_true', help="Activates the contrastive task")
parser.add_argument('--min-walltime', default=169200, type=int, help="Minimum walltime in seconds. Default: 169200s = 47h")
parser.add_argument('--normalize', action='store_true', help="Normalizes in the acceptance range")
parser.add_argument('--disable-baseline', action='store_true', help="Disables baseline distribution")
parser.add_argument('--log-dir', default='lightning_logs/', type=str, help='Logging directory. Default: \"lightning_logs/\"')
parser.add_argument('--log-exp-name', default='hparam_random_search', type=str, help='Logging experiment name. If empty, this structure level is omitted. Default: \"hparam_random_search\"')
args = parser.parse_args()


# cf. https://gist.github.com/laszukdawid/62656cf7b34cac35b325ba21d46ecfcd
# cf. https://stackoverflow.com/a/58314091
def convert_tb_data(root_dir, sort_by='step', df_names=[]):
    """Convert local TensorBoard data into a dict of Pandas DataFrames.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        Dict of pandas.DataFrames with [wall_time, step, value] columns, keyed by name.
    
    """

    def convert_tfevent(filepath):
        serialized_examples = TFRecordDataset(filepath)
        parsed_events = []
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            if len(event.summary.value) and event.summary.value[0].tag in df_names:
                parsed_events.append(parse_tfevent(event))
                
        return pd.DataFrame(parsed_events)

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )
    
    columns_order = ['wall_time', 'name', 'step', 'value']
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    all_df = {k: v for k, v in all_df.groupby('name')}
    if sort_by is not None:
        for name, df in all_df.items():
            all_df[name] = df.sort_values(sort_by).reset_index(drop=True)
    
    return all_df


def get_task_data():
    task_name = "t_"
    df_names = []
    if args.task_inpainting and not args.task_jigsaw and not args.task_contrastive:
        task_name += "inpainting__"
        df_names.append("inpainting_training_acc_epoch")
        df_names.append("inpainting_validation_acc")
    elif not args.task_inpainting and args.task_jigsaw and not args.task_contrastive:
        task_name += "jigsaw__"
        df_names.append("jigsaw_training_acc_epoch")
        df_names.append("jigsaw_validation_acc")
    elif args.task_inpainting and args.task_jigsaw and not args.task_contrastive:
        task_name += "inpainting+jigsaw__"
        df_names.append("inpainting_training_acc_epoch")
        df_names.append("inpainting_validation_acc")
        df_names.append("jigsaw_training_acc_epoch")
        df_names.append("jigsaw_validation_acc")
    elif args.task_inpainting and args.task_jigsaw and args.task_contrastive:
        task_name += "inpainting+jigsaw+contrastive__"
        df_names.append("inpainting_training_acc_epoch")
        df_names.append("inpainting_validation_acc")
        df_names.append("jigsaw_training_acc_epoch")
        df_names.append("jigsaw_validation_acc")
    else:
        raise ValueError("Invalid task configuration!")
    
    return task_name, df_names


def parse_hparam(filename):
    hparam_str = "nb_" + filename.split('nb_')[-1]
    hparam_str = hparam_str.split('__rng_')[0]
    hparam_str_partitions = hparam_str.split('__')
    hparam = {}
    for hparam_str_partition in hparam_str_partitions:
        k, v = hparam_str_partition.split('_')
        v = v.replace('-', '.')
        if k == 'inp':
            hparam[k] = v
        else:
            v = float(v)
            if v.is_integer():
                hparam[k] = int(v)
            else:
                hparam[k] = v
    
    return hparam
        

def parse():
    task_name, df_names = get_task_data()
    root_path = os.path.join(args.log_dir, args.log_exp_name, task_name)
    filenames = glob.glob("%s*" % root_path)
    num_filenames = len(filenames)
    data = {df_name: [] for df_name in df_names}
    
    for idx, filename in enumerate(filenames):
        hparam = parse_hparam(filename)
        for df_name in data.keys():
            data[df_name].append(hparam.copy())
        
        all_df = convert_tb_data(filename, df_names=df_names)
        for name, df in all_df.items():
            first_row, last_row = df.iloc[0], df.iloc[-1]
            walltime = last_row['wall_time'] - first_row['wall_time']
            if walltime < args.min_walltime:
                break
            data[name][-1]['value'] = last_row['value']
        
        print("%d / %d" % (idx+1, num_filenames))
    
    with open('%s.pkl' % task_name, 'wb') as f:
        pickle.dump(data, f)
    

if __name__ == '__main__':
    parse()