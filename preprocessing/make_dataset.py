import os
from pytorch.config import get_data_dir
import glob
import os.path as osp
import numpy as np
import scipy.io as sio
from imageio import imread

"""
File responsible for preprocessing image files to csv files: testdata.csv and traindata.csv
"""


def iterate_image_files(folder_path):
    """
    Iterates through a folder, listing any image file and produces an 'X' array with images unpacked to a 1D vector,
    and a 'Y' array with the names of t

    :param folder_path: Complete system path of folder containing the images
    :return: np.array of shape (n, 4) where n is the number of images with the given label
    """
    to_iterate = folder_path + "/*.png"
    file_iterator = glob.iglob(to_iterate)

    image_vector = []
    file_names = []

    # break_point = 0
    for full_filepath in file_iterator:
        # break_point += 1
        # if break_point >= 100:
        #     break
        img = imread(full_filepath, as_gray=True)

        unpacked = img.reshape((32*32))
        normalized = np.array([1 if i > 150 else 0 for i in unpacked])
        image_vector.append(normalized)

        file_names.append(os.path.basename(full_filepath))

    x = np.array(image_vector)
    y = np.array(file_names)

    return x, y


def shuffle_vectors(vec2d, vec1d):
    assert vec2d.shape[0] == vec1d.shape[0]
    p = np.random.permutation(vec2d.shape[0])
    return vec2d[p, :], vec1d[p]


def main(image_path, data_name, tt_split_ratio=0.8):
    datadir = get_data_dir(data_name)

    x, y = iterate_image_files(folder_path=image_path)
    x_shuffled, y_shuffled = shuffle_vectors(x, y)
    assert x_shuffled.shape == x.shape
    assert y_shuffled.shape == y.shape
    tt_split = int(tt_split_ratio*x_shuffled.shape[0])

    sio.savemat(
        file_name=osp.join(datadir, 'traindata.mat'),
        mdict={'X': x_shuffled[:tt_split, :],
               'Y': y_shuffled[:tt_split]},
    )

    sio.savemat(
        file_name=osp.join(datadir, 'testdata.mat'),
        mdict={'X': x_shuffled[tt_split:, :],
               'Y': y_shuffled[tt_split:]},
    )


if __name__ == '__main__':
    train_test_split_ratio = 0.8
    dn = "child_poet"
    base_path = r'/uio/hume/student-u31/eirikolb/img/poet_dec2_168h'
    image_folder_path = base_path + '/img_files'
    main(image_path=image_folder_path, data_name=dn, tt_split_ratio=train_test_split_ratio)
