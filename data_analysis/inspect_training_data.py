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
parser.add_argument('--data', dest='db', type=str, default='child_poet',
                    help='Name of the dataset. The name should match with the output folder name.')

"""
Utility file for inspecting some of the output and input of DCC for childpoet.
Used during development for whatever I needed in the moment, so it might seem somewhat nonsensical. 
Still, I leave it here in case others find it useful.  
"""


def inspect_class_0(arg):
    """
    Class 0 is the flat terrain for childpoet dataset.
    Therefore, we can somewhat reasonably inspect the performance of this class,
    and calculate false positives and false negatives for the class,
    which we then display as images to visually understand performance.
    Parameters
    ----------
    arg :   argparser
            Argparser that is a relic from the rest of DCC. Only has the db parameter.

    Returns
    -------
    None
    """
    # k = '10'
    # k = '15'
    # k = '20'
    # k = '25'
    # k = '30'
    k = '50'
    lr = '0_1'

    datadir = get_data_dir(arg.db)

    clustering = sio.loadmat(os.path.join(datadir, 'results/features_k{}_lr{}'.format(k, lr)))
    clustering = clustering['cluster'][0].astype(np.int)

    traindata = sio.loadmat(os.path.join(datadir, 'traindata.mat'))
    testdata = sio.loadmat(os.path.join(datadir, 'testdata.mat'))
    fulldata = np.concatenate((traindata['X'][:].astype(np.float32), testdata['X'][:].astype(np.float32)), axis=0)

    cluster_to_show = 0
    flat_terrain = np.zeros((32*32))
    flat_terrain[32*16:] = np.ones((32*16))

    print("Showing false positives for class 0")
    count = 0
    fig = plt.figure(figsize=(100, 100))
    for label, img_row in zip(clustering, fulldata):
        if label == cluster_to_show:
            if not np.array_equal(img_row, flat_terrain):

                img = img_row.reshape((32, 32))

                fig.add_subplot(4, 6, (count % 24) + 1)
                plt.imshow(img)

                count += 1
                if count == 24:
                    count = 0
                    plt.show()
                    fig = plt.figure(figsize=(100, 100))

    # make sure to show final bit too
    plt.show()
    plt.close()

    print("Showing false negatives for class 0")
    count = 0
    fig = plt.figure(figsize=(100, 100))
    for label, img_row in zip(clustering, fulldata):
        if label != cluster_to_show:
            if np.array_equal(img_row, flat_terrain):

                img = img_row.reshape((32, 32))

                fig.add_subplot(4, 6, (count % 24) + 1)
                plt.imshow(img)

                count += 1
                if count == 24:
                    count = 0
                    plt.show()
                    fig = plt.figure(figsize=(100, 100))

    # make sure to show final bit too
    plt.show()
    plt.close()


def inspect_clustering(arg):
    """
    Inspect clustering from DCC, by prints of dataset information,
    and plots of images belonging to the classes.
    In the function are parameters for thresholds of cluster sizes and similar.

    Parameters
    ----------
    arg :   argparser
            Argparser that is a relic from the rest of DCC. Only has the db parameter.

    Returns
    -------
    None
    """
    # k = '10'
    # k = '15'
    # k = '20'
    # k = '25'
    k = '30'
    # k = '50'
    lr = '0_1'

    datadir = get_data_dir(arg.db)

    clustering = sio.loadmat(os.path.join(datadir, 'results/features_k{}_lr{}'.format(k, lr)))
    traindata = sio.loadmat(os.path.join(datadir, 'traindata.mat'))
    testdata = sio.loadmat(os.path.join(datadir, 'testdata.mat'))

    clustering = clustering['cluster'][0].astype(np.int)
    fulldata = np.concatenate((traindata['X'][:].astype(np.float32), testdata['X'][:].astype(np.float32)), axis=0)

    print(len(clustering), len(fulldata), len(traindata), len(testdata), len(traindata) + len(testdata))
    print(
        clustering.shape,
        fulldata.shape,
        traindata['X'][:].astype(np.float32).shape,
        testdata['X'][:].astype(np.float32).shape
    )

    count = collections.Counter(clustering)

    threshold = 30
    threshold_type = "above"
    class_offset = 1

    if threshold_type == "none":
        to_show = [(k, count[k]) for k in count]
        print("Number of clusters:", max(clustering+1), len(count))
        print("Size of cluster 0:", to_show[0][1])
        print("Second largest cluster:", max([b for a, b in to_show[1:]]))
        print("Number of clusters total", len(to_show))
    elif threshold_type == "below":
        to_show = [(k, count[k]) for k in count if count[k] <= threshold]
        print(to_show)
        print("Number of clusters below or equal to threshold {}:".format(threshold), len(to_show))
    elif threshold_type == "above":
        to_show = [(k, count[k]) for k in count if count[k] > threshold]
        print(to_show)
        print(threshold)
        print("Number of clusters above threshold {}:".format(threshold), len(to_show))

    # return # if plotting is not desired
    for (cluster_to_show, cluster_size) in to_show[class_offset:]:
        print("Imshowing", cluster_to_show, " | Size:", cluster_size)
        count = 0
        fig = plt.figure(figsize=(100, 100))
        for cluster, img_row in zip(clustering, fulldata):
            if cluster != cluster_to_show:
                continue

            img = img_row.reshape((32, 32))

            fig.add_subplot(4, 6, count + 1)
            plt.imshow(img)
            count += 1
            if count == 24:
                count = 0
                plt.show()
                fig = plt.figure(figsize=(100, 100))

        # make sure to show final bit too
        plt.show()
        plt.close()


def main(arg):
    inspect_clustering(arg)
    # inspect_class_0(arg)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


