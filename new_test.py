import torch
import random
import numpy as np
import os

if __name__ == '__main__':

    test_npy = np.load('/root/autodl-tmp/Datasets/AMU_CSCC/Feats_Non_PE/Train/Feats/1.npy',
            allow_pickle=True)
    print(test_npy.shape)