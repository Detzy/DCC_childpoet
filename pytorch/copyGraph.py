from __future__ import print_function
import os
import h5py
import numpy as np
import argparse
import scipy.io as sio
from config import get_data_dir

# python 3 compatibility
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Note that just like in RCC & RCC-DR, the graph is built on original data.
# Once the features are extracted from the pretrained SDAE,
# they are merged along with the mkNN graph data into a single file using this module.
parser = argparse.ArgumentParser(
    description='This module is used to merge graph and extracted features into single file')
parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')
parser.add_argument('--graph', dest='g', help='path to the graph file', default=None, type=str)
parser.add_argument('--features', dest='feat', help='path to the feature file', default=None, type=str)
parser.add_argument('--out', dest='out', help='path to the output file', default=None, type=str)
parser.add_argument('--dtype', dest='dtype', type=str, default='mat', help='to store as "dtype" file')


def main(args):
    datadir = get_data_dir(args.db)

    featurefile = os.path.join(datadir, args.feat)
    graphfile = os.path.join(datadir, args.g)
    outputfile = os.path.join(datadir, args.out)
    if os.path.isfile(featurefile) and os.path.isfile(graphfile):

        if args.dtype == "h5":
            data0 = h5py.File(featurefile, 'r')
            data1 = h5py.File(graphfile, 'r')
            data2 = h5py.File(outputfile + '.h5', 'w')
        elif args.dtype == "csv":
            raise NotImplementedError
        elif args.dtype == "mat":
            fo = open(featurefile, 'rb')
            data0 = pickle.load(fo)
            data1 = sio.loadmat(graphfile)
            fo.close()
        else:
            raise ValueError("Bad file type:", args.dtype, " (Use either h5, csv or mat)")

        x0 = data0['data'][:].astype(np.float32).reshape((len(data0['labels'][:]), -1))
        x1 = data1['X'][:].astype(np.float32).reshape((len(data1['gtlabels'].T), -1))

        a, b = np.where(x0 - x1)
        # print(x0[a, b], x1[a, b])
        # c = (x0-x1).reshape(51212288)
        # print(a.size, b.size, (x0-x1).shape, sum(c))
        # print(x0-x1)
        # print(a[:5], b[:5])
        # print(c)
        assert not a.size

        joined_data = {'gtlabels': data0['labels'][:], 'X': data0['data'][:].astype(np.float32),
                       'Z': data0['Z'][:].astype(np.float32),
                       'w': data1['w'][:].astype(np.float32)}

        if args.dtype == "h5":
            data2.create_dataset('gtlabels', data=data0['labels'][:])
            data2.create_dataset('X', data=data0['data'][:].astype(np.float32))
            data2.create_dataset('Z', data=data0['Z'][:].astype(np.float32))
            data2.create_dataset('w', data=data1['w'][:].astype(np.float32))
            data0.close()
            data1.close()
            data2.close()
        elif args.dtype == "csv":
            raise NotImplementedError
        elif args.dtype == "mat":
            sio.savemat(outputfile + '.mat', joined_data)
        else:
            raise ValueError("Bad file type:", args.dtype, " (Use either h5, csv or mat)")
        return joined_data
    else:
        print('one or both the files not found')
        raise FileNotFoundError


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
