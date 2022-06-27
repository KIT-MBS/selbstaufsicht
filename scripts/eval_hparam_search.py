import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from parse_hparam_search import get_task_data

# Evaluation script for random-search-based hyperparameter optimization.

# DEFINE PARSER
parser = argparse.ArgumentParser(description='Selbstaufsicht Random-Search Hyperparameter Optimization - Evaluation Script')
parser.add_argument('--acc-threshold-inpainting-train', default=0.8, type=float, help="Minimum inpainting training accuracy for acceptance")
parser.add_argument('--acc-threshold-inpainting-val', default=0.8, type=float, help="Minimum inpainting validation accuracy for acceptance")
parser.add_argument('--acc-threshold-jigsaw-train', default=0.9, type=float, help="Minimum jigsaw training accuracy for acceptance")
parser.add_argument('--acc-threshold-jigsaw-val', default=0.9, type=float, help="Minimum jigsaw validation accuracy for acceptance")
parser.add_argument('--dh-bin-width', default=5, type=int, help="Bin width for dim-head grouping")
parser.add_argument('--task-inpainting', action='store_true', help="Activates the inpainting task")
parser.add_argument('--task-jigsaw', action='store_true', help="Activates the jigsaw task")
parser.add_argument('--task-contrastive', action='store_true', help="Activates the contrastive task")
parser.add_argument('--disable-normalize', action='store_true', help="Disables normalization in the acceptance/rejection range")
parser.add_argument('--disable-baseline', action='store_true', help="Disables baseline uniform distribution")
parser.add_argument('--disable-show', action='store_true', help="Disables plot view after creation")
args = parser.parse_args()

task_dict = {'t_inpainting__': "Inpainting",
             't_jigsaw__': "Jigsaw",
             't_inpainting+jigsaw__': "Inpainting + Jigsaw",
             't_inpainting+jigsaw+contrastive__': "Inpainting + Jigsaw + Contrastive"}

hparam_dict = {'nb': "num_blocks",
               'dh': "dim_per_head",
               'nh': "num_heads",
               'lr': "learning_rate",
               'dr': "dropout_p",
               'inp': "inpainting_masking",
               'jig': "jigsaw_partitions",
               'con': "contrastive_t"}


def plot(task_name, data, hparam_key):
    twin_axes = []

    def get_threshold(metric_name):
        if 'training' in metric_name and 'inpainting' in metric_name:
            return args.acc_threshold_inpainting_train
        elif 'training' in metric_name and 'jigsaw' in metric_name:
            return args.acc_threshold_jigsaw_train
        elif 'validation' in metric_name and 'inpainting' in metric_name:
            return args.acc_threshold_inpainting_val
        elif 'validation' in metric_name and 'jigsaw' in metric_name:
            return args.acc_threshold_jigsaw_val
        else:
            raise ValueError("Invalid metric name!", metric_name)

    def extend_plot_dict(plot_dict, k, v):
        if k in plot_dict:
            plot_dict[k] += v
        else:
            plot_dict[k] = v

    def sort_plot_dict(plot_dict):
        if len(plot_dict) == 0:
            return tuple(), tuple()

        lists = sorted(plot_dict.items())
        x, y = zip(*lists)

        return x, y

    def subplot(axes, row_idx, col_idx, x, y, y_n, baseline, num_total_runs, metric_name, threshold, hparam_key, bar_width=0.8, axis_title_y=1.0):
        if col_idx == 0:
            distribution_name = "Acceptance"
        elif col_idx == 1:
            distribution_name = "Rejection"
        else:
            raise ValueError("Invalid col_idx!", col_idx)

        x_types = [type(x_el) for x_el in x]
        if any([x_el_type in [float] for x_el_type in x_types]):
            plot_type = "scatter"
        elif all([x_el_type in [int, str] for x_el_type in x_types]):
            plot_type = "bar"
            x_ar = np.arange(len(x))
        else:
            raise TypeError("Invalid type(s) of x!", type(x[0]))

        if plot_type == 'scatter':
            l_1 = axes[row_idx, col_idx].scatter(x, y, color='b', label="Frequency distribution")
        elif plot_type == 'bar':
            l_1 = axes[row_idx, col_idx].bar(x_ar-bar_width/2, y, width=bar_width, color='b', label="Frequency distribution")
        else:
            raise ValueError("Invalid plot_type!", plot_type)
        axes[row_idx, col_idx].set_xlabel(hparam_dict[hparam_key])
        axes[row_idx, col_idx].set_ylabel("Number of occurrences")
        axes[row_idx, col_idx].set_title("%s | %s | Acceptance Threshold: %s\n%d / %d runs (%s)" % (metric_name, distribution_name, '{:.0%}'.format(threshold), sum(y), num_total_runs, '{:.0%}'.format(sum(y)/num_total_runs)), y=axis_title_y)
        if not args.disable_normalize:
            twin_axis = axes[row_idx, col_idx].twinx()
            if plot_type == 'scatter':
                l_2 = twin_axis.scatter(x, y_n, color='r', label="Frequency distribution (normalized, weighted)")
            elif plot_type == 'bar':
                l_2 = twin_axis.bar(x_ar+bar_width/2, y_n, width=bar_width, color='r', label="Frequency distribution (normalized, weighted)")
            else:
                raise ValueError("Invalid plot_type!", plot_type)
            twin_axis.set_ylabel("Number of occurrences\n(normalized, weighted)")
            twin_axes.append(twin_axis)

        if not args.disable_baseline:
            if plot_type == 'scatter':
                l_3, = axes[row_idx, col_idx].plot([min(x), max(x)], [baseline, baseline], color='black', linewidth=5., label="Baseline (uniform distribution)")
            elif plot_type == 'bar':
                l_3, = axes[row_idx, col_idx].plot([min(x_ar), max(x_ar)], [baseline, baseline], color='black', linewidth=5., label="Baseline (uniform distribution)")
            else:
                raise ValueError("Invalid plot_type!", plot_type)

        if plot_type == 'bar':
            axes[row_idx, col_idx].set_xticks(x_ar)
            axes[row_idx, col_idx].set_xticklabels(x, rotation=90, ha='right')
        if not args.disable_normalize:
            axes[row_idx, col_idx].set_zorder(1)
            axes[row_idx, col_idx].set_frame_on(False)
        axes[row_idx, col_idx].yaxis.set_major_locator(MaxNLocator(integer=True))

        plots = [l_1]
        if not args.disable_normalize:
            plots.append(l_2)
        if not args.disable_baseline:
            plots.append(l_3)
        return plots, [p.get_label() for p in plots]

    # plot acceptance and rejection distributions (horizontal) for a specific task and hparam over all runs, for all metrics (vertical)
    fig, axes = plt.subplots(len(data), 2)

    for row_idx, (metric_name, metric_data) in enumerate(data.items()):
        threshold = get_threshold(metric_name)

        acceptance_data = {}
        rejection_data = {}
        acceptance_data_n = {}
        rejection_data_n = {}

        for run_data in metric_data:
            hparam = run_data[hparam_key]
            if hparam_key == 'dh':
                div = hparam // args.dh_bin_width
                hparam = div * args.dh_bin_width
            metric_value = run_data['value']
            if metric_value >= threshold:
                extend_plot_dict(acceptance_data, hparam, 1)
                if not args.disable_normalize:
                    extend_plot_dict(acceptance_data_n, hparam, (metric_value - threshold) / (1 - threshold))
            else:
                extend_plot_dict(rejection_data, hparam, 1)
                if not args.disable_normalize:
                    extend_plot_dict(rejection_data_n, hparam, metric_value / threshold)

        x_accept, y_accept = sort_plot_dict(acceptance_data)
        _, y_accept_n = sort_plot_dict(acceptance_data_n)
        x_reject, y_reject = sort_plot_dict(rejection_data)
        _, y_reject_n = sort_plot_dict(rejection_data_n)

        accept_baseline = sum(y_accept) / len(y_accept)
        reject_baseline = sum(y_reject) / len(y_reject)

        axis_title_y = 1.1 if len(data) == 4 else 1.05
        bar_width = 0.8 if args.disable_normalize else 0.4
        _, _ = subplot(axes, row_idx, 0, x_accept, y_accept, y_accept_n, accept_baseline, len(metric_data), metric_name, threshold, hparam_key, bar_width=bar_width, axis_title_y=axis_title_y)
        handles, labels = subplot(axes, row_idx, 1, x_reject, y_reject, y_reject_n, reject_baseline, len(metric_data), metric_name, threshold, hparam_key, bar_width=bar_width, axis_title_y=axis_title_y)

    fig.legend(handles, labels, borderpad=1, handlelength=3, handletextpad=1, columnspacing=2, edgecolor="black",
               fancybox=False, loc='lower center', ncol=3).get_frame().set_linewidth(1)

    fig.suptitle("%s | %s" % (task_dict[task_name], hparam_dict[hparam_key]))
    fig.set_size_inches(16, 10)
    if len(data) == 4:
        plt.subplots_adjust(hspace=1.0, wspace=0.4)
    else:
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
    if not args.disable_show:
        plt.show()
    fig.savefig('%s%s.png' % (task_name, hparam_key), format='png', dpi=120, bbox_inches='tight')


def evaluate():
    task_name, _ = get_task_data(args)
    filename = '%s.pkl' % task_name

    # data structure: metric -> run -> hparam
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    for hparam_key in hparam_dict.keys():
        plot(task_name, data, hparam_key)


if __name__ == '__main__':
    evaluate()
