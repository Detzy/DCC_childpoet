from __future__ import print_function
import os
import random
import math
import numpy as np
import scipy.io as sio
import argparse
import collections
import matplotlib.pyplot as plt

from config import cfg, get_data_dir, get_output_dir, AverageMeter, remove_files_in_dir

import data_params as dp
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from custom_data import DCCPT_data, DCCFT_data, DCCSampler
from DCCLoss import DCCWeightedELoss, DCCLoss
from DCCComputation import makeDCCinp, computeHyperParams, computeObj

# used for logging to TensorBoard
from tensorboardX import SummaryWriter

try:
    import cPickle as pickle
except ImportError:
    import pickle


parser = argparse.ArgumentParser(description='PyTorch DCC Finetuning')
# parser.add_argument('--data', dest='db', type=str, default='child_poet',
#                     help='Name of the dataset. The name should match with the output folder name.')
parser.add_argument('--data', dest='db', type=str, default='child_poet',
                    help='Name of the dataset. The name should match with the output folder name.')


def generate_cluster_plot(arg):
    datadir = get_data_dir(arg.db)

    data_parameters = [
        ("10", "0_1"),
        ("15", "0_1"),
        ("20", "0_1"),
        ("25", "0_1"),
        ("30", "0_1"),
        ("50", "0_1"),
    ]
    for k, lr in data_parameters:
        clustering = sio.loadmat(os.path.join(datadir, 'results/features_k{}_lr{}'.format(k, lr)))
        clustering = clustering['cluster'][0].astype(np.int)

        cluster_sizes = collections.Counter(clustering)

        highest_bin = 500
        lowest_bin = 0

        bins = range(lowest_bin, highest_bin+5, 5) 

        cluster_size_hist = plt.hist(cluster_sizes.values(), bins=bins)
        cluster_size_hist.save()
        
        class_0_confusion_matrix = []


def inspect_class_0(arg):
    # k, date = '10', 'jan31'
    # k, date = '15', 'jan31'
    # k, date = '20', 'jan31'
    # k, date = '25', 'feb2'
    k, date = '30', 'feb2'
    lr = '0_1'

    datadir = get_data_dir(arg.db)

    clustering = sio.loadmat(os.path.join(datadir, 'results/features_{}_k{}_lr{}'.format(date, k, lr)))
    traindata = sio.loadmat(os.path.join(datadir, 'traindata.mat'))
    testdata = sio.loadmat(os.path.join(datadir, 'testdata.mat'))

    clustering = clustering['cluster'][0].astype(np.int)
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
    # k, date = '10', 'jan31'
    # k, date = '15', 'jan31'
    # k, date = '20', 'jan31'
    # k, date = '25', 'feb2'
    k, date = '30', 'feb2'
    lr = '0_1'

    datadir = get_data_dir(arg.db)

    clustering = sio.loadmat(os.path.join(datadir, 'results/features_{}_k{}_lr{}'.format(date, k, lr)))
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

    threshold = 5
    threshold_type = "none"
    class_offset = 1

    if threshold_type == "none":
        to_show = [(k, count[k]) for k in count]
        print("Number of clusters:", max(clustering+1), len(count))
        print("Size of cluster 0:", to_show[0][1])
        print("Second largest cluster:", max([b for a, b in to_show[1:]]))
        print("Number of clusters total", len(to_show))
    elif threshold_type == "below":
        to_show = [(k, count[k]) for k in count if count[k] <= threshold]
        print("Number of clusters below or equal to threshold {}:".format(threshold), len(to_show))
    elif threshold_type == "above":
        to_show = [(k, count[k]) for k in count if count[k] > threshold]
        print("Number of clusters above threshold {}:".format(threshold), len(to_show))

    return # if plotting is not desired
    for (cluster_to_show, cluster_size) in to_show[class_offset:]:
        print("Imshowing", cluster_to_show, " | Size:", cluster_size)
        count = 0
        fig = plt.figure(figsize=(100, 100))
        for cluster, img_row in zip(clustering, fulldata):
            if cluster != cluster_to_show:
                continue

            img = img_row.reshape((32, 32))

            fig.add_subplot(3, 5, count + 1)
            plt.imshow(img)
            count += 1
            if count == 15:
                count = 0
                plt.show()
                fig = plt.figure(figsize=(100, 100))

        # make sure to show final bit too
        plt.show()
        plt.close()


def inspect_files(arg):
    datadir = get_data_dir(arg.db)

    # fo = open(os.path.join(datadir, 'pretrained.pkl'), 'rb')
    # pretrained_pkl = pickle.load(fo)
    # fo.close()

    pretrained_mat = sio.loadmat(os.path.join(datadir, 'pretrained.mat'))
    train_data = sio.loadmat(os.path.join(datadir, 'traindata.mat'))
    test_data = sio.loadmat(os.path.join(datadir, 'testdata.mat'))

    # print(pretrained_pkl['Z'])
    print(pretrained_mat['X'])
    print(pretrained_mat['w'].shape)
    print(pretrained_mat['gtlabels'])
    for row in pretrained_mat['w']:
        print(row)
    # print(pretrained_mat.keys())
    # print(train_data.keys())
    # print(train_data['X'].shape)
    # print(train_data['Y'].shape)
    # print(test_data['X'].shape)
    # print(test_data['Y'].shape)

    # print(np.asarray(train_data['X']))
    # for (x, y), elm in np.ndenumerate(np.asarray(train_data['X'])):
    #     if elm != 1 and elm != 0:
    #         print(elm)
    #
    # print(np.asarray(test_data['X']))
    # for (x, y), elm in np.ndenumerate(np.asarray(test_data['X'])):
    #     if elm != 1 and elm != 0:
    #         print(elm)
    #
    # print(np.asarray(train_data['Y']))
    # for (x, y), elm in np.ndenumerate(np.asarray(train_data['Y'])):
    #     if elm not in range(0, 3):
    #         print(elm)



def main(arg):
    inspect_clustering(arg)
    # inspect_class_0(arg)
    # inspect_files(arg)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


