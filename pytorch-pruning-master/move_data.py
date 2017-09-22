import glob
import os, sys, shutil
import random
import numpy as np

np.random.seed(10000)

def moveFiles(dir_from, dir_to, fileType=None, numFilesToKeepInSrc=-1):
    # if numFiles=-1, no files will be kept in dir_from
    # if numFiles>0, random numFilesToKeepInSrc files will be kept in dir_from

    fullPaths_from = glob.glob(dir_from + '*.' + fileType)
    fullPaths_to = [dir_to + p.split('/')[-1] for p in fullPaths_from]
    l = range(len(fullPaths_from))
    random.shuffle(l)
    l = l[:len(fullPaths_from) - numFilesToKeepInSrc]
    for i in l:
        shutil.move(fullPaths_from[i], fullPaths_to[i])


cats_dir_from = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/cats/'
cats_dir_to = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/cats_extra/'
dogs_dir_from = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/dogs/'
dogs_dir_to = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/dogs_extra/'

moveFiles(cats_dir_from, cats_dir_to, fileType='jpg', numFilesToKeepInSrc=2000)
moveFiles(dogs_dir_from, dogs_dir_to, fileType='jpg', numFilesToKeepInSrc=2000)