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


    


def inspect_clustering(arg):
    datadir = get_data_dir(arg.db)

    clustering = sio.loadmat(os.path.join(datadir, 'results/features'))
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

    # fo = open(os.path.join(datadir, 'pretrained.pkl'), 'rb')
    # data2 = pickle.load(fo)
    # fo.close()

    # train_data = zip(
    #     clustering,
    #     fulldata,
    # )

    count = collections.Counter(clustering)

    threshold = 30
    above_threshold = [k for k in count if count[k] < threshold]

    print(max(clustering), min(clustering))
    print(clustering)
    print([(k, count[k]) for k in count if count[k] > 30])

    for cluster_to_show in above_threshold[1:]:
        print("Imshowing", cluster_to_show)
        count = 0
        fig = plt.figure(figsize=(100, 100))
        for cluster, img_row in zip(clustering, fulldata):
            if cluster != cluster_to_show:
                continue

            img = img_row.reshape((32, 32))

            # img1 = row1.reshape((28, 28))
            # img2 = row2.reshape((28, 28))
            # print("Imshowing", cluster)

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
    # inspect_clustering(arg)
    inspect_files(arg)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


