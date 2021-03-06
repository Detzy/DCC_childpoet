from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import argparse
import collections
import pandas as pd
import matplotlib.pyplot as plt

from config import get_data_dir

try:
    import cPickle as pickle
except ImportError:
    import pickle


parser = argparse.ArgumentParser(description='PyTorch DCC Finetuning')
parser.add_argument('--data', dest='db', type=str, default='child_poet_rebalanced',
                    help='Name of the dataset. The name should match with the output folder name.')


def generate_cluster_plots(arg):
    """
    This function is made to output useful plots and statistics of the DCC clustering of childpoet dataset.
    First output is a set of histogram plots of the cluster size distribution (with the outlier class 0 not included).
    Second output is a confusion matrix for the outlier class 0. The confusion matrix is technically meaningless
    for clustering algorithms, but makes sense if we assume that class 0 should be perfectly flat terrain.

    Parameters
    ----------
    arg :   argparser
            Argparser that is a relic from the rest of DCC. Only has the db parameter.

    Returns
    -------
    None
    """
    datadir = get_data_dir(arg.db)

    ymin = 0
    ymax = 3000

    data_parameters = [
        ("10", "0_1"),
        ("15", "0_1"),
        ("20", "0_1"),
        ("25", "0_1"),
        ("30", "0_1"),
        ("50", "0_1"),
    ]

    class_0_confusion_matrix = {
        'k': [],
        'n': [],
        'c_n': [],
        'lr': [],
        'tp': [],
        'fp': [],
        'tn': [],
        'fn': [],
    }

    for k, lr in data_parameters:
        clustering = sio.loadmat(os.path.join(datadir, 'results/features_k{}_lr{}'.format(k, lr)))
        clustering = clustering['cluster'][0].astype(np.int)

        traindata = sio.loadmat(os.path.join(datadir, 'traindata.mat'))
        testdata = sio.loadmat(os.path.join(datadir, 'testdata.mat'))
        fulldata = np.concatenate((traindata['X'][:].astype(np.float32), testdata['X'][:].astype(np.float32)), axis=0)

        # ------------------------------
        # Make cluster histogram (cluster 0 omitted, due to being extreme outlier)
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True

        cluster_sizes = collections.Counter(clustering)

        highest_bins = list(cluster_sizes.values())
        highest_bins.sort(reverse=True)
        second_highest_bin = highest_bins[1] + 1  # Selected as the SECOND largest cluster
        steps = 20

        plt.title(label="Cluster distribution for parameters k={}".format(k))
        plt.ylabel(ylabel="Number of clusters")
        plt.xlabel(xlabel="Cluster size")

        plt.hist(
            cluster_sizes.values(),
            bins=steps,
            range=(0, second_highest_bin),
            log=True
        )

        ax = plt.gca()
        ax.set_ylim([ymin, ymax])

        plt.savefig(os.path.join(datadir, f'analysis/cluster_histogram_k{k}_lr{lr}'))
        plt.clf()

        # ------------------------------
        # Create confusion matrix for class 0 (flat terrain)
        # Note that in some cases, flat terrain is not actually labeled as class 0,
        # so we need to find the actual class 0 label
        l1, l2 = zip(*sorted(zip(list(cluster_sizes.values()), list(cluster_sizes.keys())), reverse=True))
        flat_terrain_label = l2[0]
        print(flat_terrain_label)

        flat_terrain = np.zeros((32 * 32))
        flat_terrain[32 * 16:] = np.ones((32 * 16))

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for label, img_row in zip(clustering, fulldata):
            if label == flat_terrain_label:
                if np.array_equal(img_row, flat_terrain):
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if np.array_equal(img_row, flat_terrain):
                    false_negative += 1
                else:
                    true_negative += 1

        class_0_confusion_matrix['k'].append(k)
        class_0_confusion_matrix['n'].append(len(clustering))
        class_0_confusion_matrix['c_n'].append(max(clustering))
        class_0_confusion_matrix['lr'].append(lr)
        class_0_confusion_matrix['tp'].append(true_positive)
        class_0_confusion_matrix['fp'].append(false_positive)
        class_0_confusion_matrix['tn'].append(true_negative)
        class_0_confusion_matrix['fn'].append(false_negative)

        df = pd.DataFrame(class_0_confusion_matrix)
        df.to_csv(os.path.join(datadir, f'analysis/class_0_confusion_matrix_k{k}_lr{lr}'))


def main(arg):
    generate_cluster_plots(arg)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


