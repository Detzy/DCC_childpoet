from __future__ import print_function
import os
import random
import math
import numpy as np
import scipy.io as sio
import argparse

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

"""
IMPORTANT! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!
THIS FILE DOES NOT WORK!
IT IS NOT YET IMPLEMENTED, AND LIKELY NEVER WILL BE!
IF IT STILL EXISTS, THIS IS PURELY BECAUSE I FORGOT TO DELETE IT
!!!!!!!!!!!!!!!!!!!!!!!!!!! 
"""

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch DCC Finetuning')
parser.add_argument('--data', dest='db', type=str, default='child_poet',
                    help='Name of the dataset. The name should match with the output folder name.')
parser.add_argument('--batchsize', type=int, default=1, help='batch size used for Finetuning')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--net', dest='torchmodel', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--dtype', dest='dtype', type=str, default='mat', help='to store as "dtype" file')
parser.add_argument('--dim', type=int, help='dimension of embedding space', default=10)
# parser.add_argument('--nepoch', type=int, default=500, help='maximum number of iterations used for Finetuning')
# By default M = 20 is used. For convolutional SDAE M=10 was used.
# Similarly, for different NW architecture different value for M may be required.
# parser.add_argument('--M', type=int, default=20, help='inner number of epochs at which to change lambda')
# parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=480, type=int, help='epoch to resume from')
# parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
# parser.add_argument('--id', type=int, help='identifying number for storing tensorboard logs')
# parser.add_argument('--clean_log', action='store_true', help='remove previous tensorboard logs under this ID')


def main(args, net=None):
    raise NotImplementedError

    datadir = get_data_dir(args.db)
    outputdir = get_output_dir(args.db)

    use_cuda = torch.cuda.is_available()

    # Set the seed for reproducing the results
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # setting up dataset specific objects
    single_set = DCCPT_data(root=datadir+'_single', train=False, dtype=args.dtype, single_data_point=True)
    numeval = 1
    # assert len(single_set) == 1

    # extracting training data from the pretrained.mat file
    data, labels, pairs, Z, sampweight = makeDCCinp(args)
    print(pairs)

    # For simplicity, I have created placeholder for each datasets and model
    load_pretraining = True if net is None else False
    if net is None:
        net = dp.load_predefined_extract_net(args)

    # computing and initializing the hyperparams
    _sigma1, _sigma2, _lambda, _delta, _delta1, _delta2, lmdb, lmdb_data = computeHyperParams(pairs, Z)

    # Create dataset and random batch sampler for Finetuning stage
    trainset = DCCFT_data(pairs, data, sampweight)
    batch_sampler = DCCSampler(trainset, shuffle=True, batch_size=args.batchsize)

    # copying model params from Pretrained (SDAE) weights file
    if load_pretraining:
        load_weights(args, outputdir, net)


    # creating objects for loss functions, U's are initialized to Z here
    # Criterion1 corresponds to reconstruction loss
    criterion1 = DCCWeightedELoss(size_average=True)
    # Criterion2 corresponds to sum of pairwise and data loss terms
    criterion2 = DCCLoss(Z.shape[0], Z.shape[1], Z, size_average=True)

    if use_cuda:
        net.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()

    # setting up data loader for classification phase
    single_file_loader = torch.utils.data.DataLoader(single_set, batch_size=args.batchsize, shuffle=False, **kwargs)

    # setting up optimizer - the bias params should have twice the learning rate w.r.t. weights params
    # bias_params = filter(lambda x: ('bias' in x[0]), net.named_parameters())
    # bias_params = list(map(lambda x: x[1], bias_params))
    # nonbias_params = filter(lambda x: ('bias' not in x[0]), net.named_parameters())
    # nonbias_params = list(map(lambda x: x[1], nonbias_params))
    #
    # optimizer = optim.Adam([{'params': bias_params, 'lr': 2*args.lr},
    #                         {'params': nonbias_params},
    #                         {'params': criterion2.parameters(), 'lr': args.lr},
    #                         ], lr=args.lr, betas=(0.99, 0.999))

    # this is needed for WARM START
    filename = outputdir+'/FTcheckpoint_%d.pth.tar' % args.level
    if os.path.isfile(filename):
        print("==> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint['state_dict'])
        criterion2.load_state_dict(checkpoint['criterion_state_dict'])
        # startepoch = checkpoint['epoch']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        _sigma1 = checkpoint['sigma1']
        _sigma2 = checkpoint['sigma2']
        _lambda = checkpoint['lambda']
        _delta = checkpoint['delta']
        _delta1 = checkpoint['delta1']
        _delta2 = checkpoint['delta2']
    else:
        print("==> no checkpoint found at '{}'".format(filename))
        raise ValueError

    # This is the actual Algorithm
    Z, U, change_in_assign, assignment = classify(single_file_loader, net, criterion2, use_cuda, _delta, numeval, pairs)

    output = {'Z': Z, 'U': U, 'gtlabels': labels, 'w': pairs, 'cluster': assignment}
    print(max(assignment))
    print(output)
    # sio.savemat(os.path.join(outputdir, 'features'), output)


def load_weights(args, outputdir, net):
    filename = os.path.join(outputdir, args.torchmodel)
    if os.path.isfile(filename):
        print("==> loading params from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print("==> no checkpoint found at '{}'".format(filename))
        raise ValueError


# This method is called "test" in standard DCC
def classify(testloader, net, criterion, use_cuda, _delta, numeval, pairs):
    net.eval()

    original = []
    features = []
    labels = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs_Var = Variable(inputs, volatile=True)
        enc, dec = net(inputs_Var)
        features += list(enc.data.cpu().numpy())
        labels += list(targets)
        original += list(inputs.cpu().numpy())
        break

    original, features, labels = np.asarray(original).astype(np.float32), np.asarray(features).astype(np.float32), \
                                 np.asarray(labels)

    U = criterion.U.data.cpu().numpy()

    change_in_assign = 0
    assignment = -np.ones(len(labels))
    index, ari, ami, nmi, acc, n_components, assignment = computeObj(U, pairs, _delta, labels, numeval)

    return features, U, change_in_assign, assignment


def plot_to_image(U, title):
    plt.clf()
    plt.scatter(U[:, 0], U[:, 1])
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    # image = ToTensor()(image).unsqueeze(0)
    image = ToTensor()(image)
    return image


# Saving checkpoint
def save_checkpoint(state, index, filename):
    newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
    torch.save(state, newfilename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
