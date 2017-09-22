import glob
import os, sys, shutil
import random

def moveFiles(dir_from, dir_to, numFiles=-1):
    # if numFiles=-1, all files will be moved
    # if numFiles>0, random numFiles files will be moved

    fullPaths_from = glob.glob(dir_from)
    fullPaths_to = [dir_to + p.split('/')[-1] for p in fullPaths_from]
    l = random.shuffle(range(len(fullPaths_from)))
    for i in l:
        shutil.move(fullPaths_from[i], fullPaths_to[i])


cats_dir_from = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/cats/'
cats_dir_to = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/cats_extra/'
dogs_dir_from = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/dogs/'
dogs_dir_to = '/home/vijetha/Dropbox (ASU)/Pruning_Retraining/pytorch-pruning-master/train/dogs_extra/'

moveFiles(cats_dir_from, cats_dir_to)
moveFiles(dogs_dir_from, dogs_dir_to)
