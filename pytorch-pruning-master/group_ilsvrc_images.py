import glob
import os, sys, shutil
import random
import scipy.io as sio

# this code will categorize validation images into different folders - one folder per label. Number of images in each
# folder is set using the "nImagesPerCat" argument


img_path = '/data1/ImageNet_Fall2011/ILSVRC2012_val_images/'
grouped_imgs_path = '/data1/ImageNet_Fall2011/ILSVRC2012_grouped/'
full_paths = glob.glob(img_path + '*.JPEG')
full_paths = sorted(full_paths, key=lambda name: int(name[62:70]))  # important
label_file = '/data1/ImageNet_Fall2011/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
synset_to_word_file = '/data1/ImageNet_Fall2011/ILSVRC2012_devkit_t12/data/synset_to_word_ilsvrc2012.txt'
ilsvrc_val_labels = '/data1/ImageNet_Fall2011/ILSVRC2012_devkit_t12/data/ILSVRC2012_Caffe_val_labels.txt'

nImagesPerCat = 5

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
    if (not os.path.exists(grouped_imgs_path + str(i).zfill(4) + '_' + l)):
        os.makedirs(grouped_imgs_path + str(i).zfill(4) + '_' + l)

synsets = sio.loadmat('/data1/ImageNet_Fall2011/ILSVRC2012_devkit_t12/data/meta.mat')
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

for key in grouped_full_paths:
    random.shuffle(grouped_full_paths[key])
    grouped_full_paths[key] = grouped_full_paths[key][:nImagesPerCat]

for key in grouped_full_paths:
    for p in grouped_full_paths[key]:
        imgName = p.split('/')[-1]
        shutil.copy(p, grouped_imgs_path + key + '/' + imgName)

