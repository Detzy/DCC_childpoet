"""Example for doing all steps in code only (other examples require calling different files separately)"""
import torch.nn as nn

from config import cfg, get_data_dir
from easydict import EasyDict as edict
from edgeConstruction import compressed_data
import matplotlib.pyplot as plt
import data_params as dp
import pretraining
import extract_feature
import copyGraph
import DCC


class IdentityNet(nn.Module):
    """Substitute for the autoencoder for visualization and debugging just the clustering part"""
    def __init__(self):
        super(IdentityNet, self).__init__()

    def forward(self, x):
        # internal encoding is x and output is also just x
        return x, x


dataset_name = "child_poet"

datadir = get_data_dir(dataset_name)
N = 600

# unlike in the easy example, we cant visualize the data because we don't know what optimal classes should be

# So instead, we go straight to: construct mkNN graph
k = 50
compressed_data(dataset_name, N, k, preprocess='none', algo='knn', isPCA=None, format='mat')

# then pretrain to get features
args = edict()
args.db = dataset_name
args.niter = 500
args.step = 300
args.lr = 0.001

# if we need to resume for faster debugging/results
args.resume = False
args.level = None

args.batchsize = 300
args.ngpu = 1
args.deviceID = 0
args.tensorboard = True
args.dtype = "mat"
args.id = 2
args.dim = 2
args.manualSeed = cfg.RNG_SEED
args.clean_log = True

# if we comment out the next pretraining step and the identity network, use the latest checkpoint
index = len(dp.child_poet.dim) - 1
net = None
# if we comment out the next pretraining step we use the identity network
net = IdentityNet()
index, net = pretraining.main(args)

# extract pretrained features
args.feat = 'pretrained'
args.torchmodel = 'checkpoint_{}.pth.tar'.format(index)
extract_feature.main(args, net=net)

# merge the features and mkNN graph
args.g = 'pretrained.mat'
args.out = 'pretrained'
args.feat = 'pretrained.pkl'
copyGraph.main(args)

# actually do DCC
args.batchsize = cfg.PAIRS_PER_BATCH
args.nepoch = 500
args.M = 20
args.lr = 0.001
out = DCC.main(args, net=net)
