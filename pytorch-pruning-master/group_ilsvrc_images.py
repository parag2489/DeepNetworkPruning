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
nImagesPerCat = 5

file = open(label_file, 'r')

labels = []
for line in file:
    labels.append(int(line))

synsets = sio.loadmat('/data1/ImageNet_Fall2011/ILSVRC2012_devkit_t12/data/meta.mat')
synsets = synsets['synsets']

wordLabelMapping = {}
for s in synsets:
    assert len(s['ILSVRC2012_ID'][0][0]) == 1
    imagenet_id = s['ILSVRC2012_ID'][0][0][0]
    label = '_'.join(str(s['words'][0][0]).split(',')[0].split(' '))
    wordLabelMapping[imagenet_id] = label
    if not os.path.exists(grouped_imgs_path + str(imagenet_id) + '_' + label):
        os.makedirs(grouped_imgs_path + str(imagenet_id) + '_' + label)

grouped_full_paths = {}
for p, l in zip(full_paths, labels):
    if l not in grouped_full_paths:
        grouped_full_paths[l] = [p]
    else:
        grouped_full_paths[l].append(p)

for key in grouped_full_paths:
    random.shuffle(grouped_full_paths[key])
    grouped_full_paths[key] = grouped_full_paths[key][:nImagesPerCat]

for key in grouped_full_paths:
    for p in grouped_full_paths[key]:
        imgName = p.split('/')[-1]
        shutil.copy(p, grouped_imgs_path + str(key) + '_' + wordLabelMapping[key] + '/' + imgName)

