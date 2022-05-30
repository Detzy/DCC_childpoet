from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import argparse
import collections
import matplotlib.pyplot as plt

from config import get_data_dir

try:
    import cPickle as pickle
except ImportError:
    import pickle


parser = argparse.ArgumentParser(description='PyTorch DCC Finetuning')
parser.add_argument('--data', dest='db', type=str, default='child_poet_rebalanced',
                    help='Name of the dataset. The name should match with the output folder name.')

"""
"""


def save_to_file(figure, cluster_label, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    figure.savefig(output_folder_path + "/class{}.png".format(cluster_label))


def plot_clustering_samples(clustering, fulldata, threshold, class_samples, output_folder_path):

    count = collections.Counter(clustering)
    to_plot_from = [k for k in count if count[k] >= threshold]
    print("Number of clusters above threshold {}:".format(threshold), len(to_plot_from))

    for cluster_to_plot in to_plot_from:
        count = 0
        fig = plt.figure()
        # fig = plt.figure(figsize=(10, 10))
        fig.suptitle("Samples from class {}".format(cluster_to_plot))
        for cluster, img_row in zip(clustering, fulldata):
            if cluster != cluster_to_plot:
                continue

            img = img_row.reshape((32, 32))

            fig.add_subplot(3, 4, count + 1)
            plt.imshow(img)
            count += 1
            if count == class_samples:
                save_to_file(fig, cluster_to_plot, output_folder_path)
                plt.close(fig)
                break


def main(arg):
    k = '10'
    # k = '15'
    # k = '20'
    # k = '25'
    # k = '30'
    # k = '50'
    lr = '0_1'
    threshold = 30
    class_samples = 10

    datadir = get_data_dir(arg.db)
    out_path = datadir+"/analysis/class_comparison_k{}_lr{}".format(k, lr)

    clustering = sio.loadmat(os.path.join(datadir, 'results/features_k{}_lr{}'.format(k, lr)))
    traindata = sio.loadmat(os.path.join(datadir, 'traindata.mat'))
    testdata = sio.loadmat(os.path.join(datadir, 'testdata.mat'))

    clustering = clustering['cluster'][0].astype(np.int)
    fulldata = np.concatenate((traindata['X'][:].astype(np.float32), testdata['X'][:].astype(np.float32)), axis=0)

    plot_clustering_samples(clustering, fulldata, threshold, class_samples, out_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


