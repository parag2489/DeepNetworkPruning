import glob
import os, sys, shutil
import random
import scipy.io as sio

# this code will categorize validation images into small train and test. The train and test folders will in turn
# have different folders - one folder per label. Number of images in each folder in test is set using the
# "nImagesPerCat_te" argument

root_path = '/data1/VisionDatasets/ImageNet_Fall2011/'
img_path = root_path + 'ILSVRC2012_val_images/'
grouped_imgs_path_tr = root_path + 'ILSVRC2012_grouped/train/'
grouped_imgs_path_te = root_path + 'ILSVRC2012_grouped/test/'
full_paths = glob.glob(img_path + '*.JPEG')
full_paths = sorted(full_paths, key=lambda name: int(name[77:85]))  # important
label_file = root_path + 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
synset_to_word_file = root_path + 'ILSVRC2012_devkit_t12/data/synset_to_word_ilsvrc2012.txt'
ilsvrc_val_labels = root_path + 'ILSVRC2012_devkit_t12/data/ILSVRC2012_Caffe_val_labels.txt'

nImagesPerCat_te = 5

file = open(label_file, 'r')

labels = []
for line in file:
    labels.append(int(line))

file = open(synset_to_word_file, 'r')
synset_to_word_mapping = {}

for i, line in enumerate(file):
    syn = line.split(' ')[0]
    l = '_'.join(line.split(' ', 1)[1].split(',')[0].split(' ')).strip('\n')
    synset_to_word_mapping[syn] = (str(i).zfill(4), l)
    if (not os.path.exists(grouped_imgs_path_tr + str(i).zfill(4) + '_' + l)):
        os.makedirs(grouped_imgs_path_tr + str(i).zfill(4) + '_' + l)
    if (not os.path.exists(grouped_imgs_path_te + str(i).zfill(4) + '_' + l)):
        os.makedirs(grouped_imgs_path_te + str(i).zfill(4) + '_' + l)

synsets = sio.loadmat(root_path + 'ILSVRC2012_devkit_t12/data/meta.mat')
synsets = synsets['synsets']

label_to_synset_mapping = {}
for s in synsets:
    assert len(s['ILSVRC2012_ID'][0][0]) == 1
    imagenet_id = s['ILSVRC2012_ID'][0][0][0]
    wnid = str(s['WNID'][0][0])
    label_to_synset_mapping[imagenet_id] = wnid

grouped_full_paths = {}
for p, l in zip(full_paths, labels):
    curr_wnid = label_to_synset_mapping[l]
    curr_val_label = str(synset_to_word_mapping[curr_wnid][0]) + '_' + synset_to_word_mapping[curr_wnid][1]
    if curr_val_label not in grouped_full_paths:
        grouped_full_paths[curr_val_label] = [p]
    else:
        grouped_full_paths[curr_val_label].append(p)

grouped_full_paths_tr = {}
grouped_full_paths_te = {}

for key in grouped_full_paths:
    random.shuffle(grouped_full_paths[key])
    grouped_full_paths_te[key] = grouped_full_paths[key][:nImagesPerCat_te]
    grouped_full_paths_tr[key] = grouped_full_paths[key][nImagesPerCat_te:]

for key in grouped_full_paths_tr:
    for p in grouped_full_paths_tr[key]:
        imgName = p.split('/')[-1]
        shutil.copy(p, grouped_imgs_path_tr + key + '/' + imgName)

for key in grouped_full_paths_te:
    for p in grouped_full_paths_te[key]:
        imgName = p.split('/')[-1]
        shutil.copy(p, grouped_imgs_path_te + key + '/' + imgName)

