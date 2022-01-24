from __future__ import print_function
import os
import random
import math
import numpy as np
import scipy.io as sio
import argparse
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
parser.add_argument('--data', dest='db', type=str, default='mnist',
                    help='Name of the dataset. The name should match with the output folder name.')


def main(arg):
    threshold = 100
    # outputdir = get_output_dir(arg.db)
    datadir = get_data_dir(arg.db)

    data1 = sio.loadmat(os.path.join(datadir, 'pretrained.mat'))

    fo = open(os.path.join(datadir, 'pretrained.pkl'), 'rb')
    data2 = pickle.load(fo)
    fo.close()

    train_data = zip(data1['X'][:].astype(np.float32),
                     data2['data'][:].astype(np.float32))

    # train_labels = np.squeeze(data['Y'][:])
    for row1, row2 in train_data:
        # img1 = row1.reshape((32, 32))
        # img2 = row2.reshape((32, 32))
        img1 = row1.reshape((28, 28))
        img2 = row2.reshape((28, 28))
        print("Imshowing")
        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(img1)
        fig.add_subplot(2, 1, 2)
        plt.imshow(img2)
        # plt.imshow(img2)
        plt.show()
    # output = sio.loadmat(os.path.join(outputdir, 'features'))
    # labels = output.get('gtlabels')[0]
    # clusters = output.get('cluster')[0]
    #
    # print(min(labels), max(labels))
    # print(min(clusters), max(clusters))
    #
    # cumsum = 0
    # for i in range(10):
    #     cumsum += sum(labels == i)
    #     print(sum(labels == i), "of", i, "| cumulative:", cumsum)
    #
    # print("----------------")
    # cumsum = 0
    # for i in range(max(clusters)+1):
    #     entries = sum(clusters == i)
    #     if entries > threshold:
    #         cumsum += entries
    #         print(entries, "of", i, "| cumulative:", cumsum)
    #
    # print(data['X'][0][:])
    # print(data['X'][0][:])



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


