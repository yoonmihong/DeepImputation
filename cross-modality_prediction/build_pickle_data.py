#!/usr/bin/Python
# -*- coding: utf-8 -*-

import os
from six.moves import cPickle as pickle
import nrrd
import numpy as np
import random
from os import listdir

"""
A:6/12/24 month
B:6/12/24 month

(A, B) is paired data you need depends on tasks
if you want to transfer 6 month to 12 month, A: 6month B: 12month

(A, B, path)
the purpose of adding path in this data tuple is to look back subject name
"""

path_data = '/Human2/ImageImputation/Data_resampled/EBDS/'
path_pickle = '/Human2/MachineLearningStudies/Imputation/ymhong/code/DeepImputation/cross-modality_prediction/pickle'
path_save = os.path.join(path_pickle, 'paired_EBDS_resampled_1year')

pair_A = 'T1'
pair_B = 'T2'


object_shape = (160,192,144)
original_shape = (195, 233, 159)


if not os.path.isdir(path_pickle):
	os.mkdir(path_pickle)

if not os.path.isdir(path_save):
	os.mkdir(path_save)

def crop(img, object_shape, crop_indices=None):
    img = img.transpose((2, 1, 0))

    # for IBIS: some data have non-zero values outside brain mask
    img_tmp = img#np.zeros(img.shape, dtype=np.float32)
    if img[0,0,0] != 0:
        img_tmp[img==img[0,0,0]] = 0
    if crop_indices is None:
        crop_indices = np.zeros(6,dtype=np.int32)
        # step 1: crop the brain from the original image
        index_nonzero = np.argwhere(img_tmp > 0)

        index_z_begin = np.min(index_nonzero[:, 0])
        index_z_end = np.max(index_nonzero[:, 0])

        index_y_begin = np.min(index_nonzero[:, 1])
        index_y_end = np.max(index_nonzero[:, 1])

        index_x_begin = np.min(index_nonzero[:, 2])
        index_x_end = np.max(index_nonzero[:, 2])
        
        crop_indices[0] = index_z_begin
        crop_indices[1] = index_z_end
        crop_indices[2] = index_y_begin
        crop_indices[3] = index_y_end
        crop_indices[4] = index_x_begin
        crop_indices[5] = index_x_end
    else:
        index_z_begin = crop_indices[0]
        index_z_end = crop_indices[1]
        index_y_begin = crop_indices[2]
        index_y_end = crop_indices[3]
        index_x_begin = crop_indices[4]
        index_x_end = crop_indices[5]

    img_crop = img[index_z_begin:index_z_end, index_y_begin:index_y_end, index_x_begin: index_x_end]

    z_crop, y_crop, x_crop = img_crop.shape

    if z_crop > object_shape[0] or y_crop > object_shape[1] or x_crop > object_shape[2]:
        print ("images are bigger than the cropping sizes")
        print (index_z_begin, index_z_end, index_y_begin, index_y_end, index_x_begin, index_x_end)
        return

    # step 2: padding
    img_padding = np.zeros(object_shape, dtype=np.float32)

    original_shape = img_crop.shape
    index_z_begin_new = int((object_shape[0] - original_shape[0]) / 2)
    index_z_end_new = index_z_begin_new + original_shape[0]

    index_y_begin_new = int((object_shape[1] - original_shape[1]) / 2)
    index_y_end_new = index_y_begin_new + original_shape[1]

    index_x_begin_new = int((object_shape[2] - original_shape[2]) / 2)
    index_x_end_new = index_x_begin_new + original_shape[2]

    img_padding[index_z_begin_new: index_z_end_new, index_y_begin_new: index_y_end_new,
    index_x_begin_new: index_x_end_new] = img_crop

    img_padding = img_padding.transpose((2, 1, 0))

    return img_padding, crop_indices

def save_pickle(path_data, save_path):
	for t1_file_path in path_data:
		t2_file_path = t1_file_path.replace('T1', 'T2').replace('t1', 't2')

		img_T1, header = nrrd.read(t1_file_path)
		img_T1 = img_T1.transpose((2, 1, 0))
		img_T1[img_T1<0.0] = 0.0
		crop_indices = [int((original_shape[0] - object_shape[0])/2),int((original_shape[0] - object_shape[0])/2)+object_shape[0],
					int((original_shape[1] - object_shape[1])/2),int((original_shape[1] - object_shape[1])/2)+object_shape[1],
					int((original_shape[2] - object_shape[2])/2),int((original_shape[2] - object_shape[2])/2)+object_shape[2]
		]
		img_T1, _ = crop(img_T1, object_shape, crop_indices)
		# print (crop_indices)
		img_T2, header = nrrd.read(t2_file_path)
		img_T2 = img_T2.transpose((2, 1, 0))
		img_T2[img_T2<0.0] = 0.0
		img_T2,_ = crop(img_T2, object_shape, crop_indices)
		if img_T1 is not None and img_T2 is not None:

			# img_T1, _ = normalization(img_T1)
			min_val_T1 = np.percentile(img_T1, 1)
			max_val_T1 = np.percentile(img_T1, 99)
			img_T1 = (img_T1 - min_val_T1) / (max_val_T1 - min_val_T1)
			img_T1[img_T1 < 0] = 0
			img_T1[img_T1 > 1] = 1
			print (min_val_T1, max_val_T1)
			img_T1 = np.expand_dims(img_T1, axis=-1)

			# img_T2, _ = normalization(img_T2)
			min_val_T2 = np.percentile(img_T2, 1)
			max_val_T2 = np.percentile(img_T2, 99)
			img_T2 = (img_T2 - min_val_T2) / (max_val_T2 - min_val_T2)
			img_T2[img_T2 < 0] = 0
			img_T2[img_T2 > 1] = 1
			print (min_val_T2, max_val_T2)
			img_T2 = np.expand_dims(img_T2, axis=-1)

			save_file_name = os.path.split(t1_file_path)[1]
			save_file_name = os.path.splitext(save_file_name)[0]
			save_file_name = save_file_name.replace('T1', 't1').replace('t1', 't1_t2')
			pickle_save_path = os.path.join(save_path, save_file_name + '.pkl')

			# print(pickle_save_path)
			with open(pickle_save_path, 'wb') as f:
				# print(f'saving {pickle_save_path}')
				pickle.dump([img_T1, img_T2, os.path.split(t1_file_path)[1], min_val_T1, max_val_T1, min_val_T2, max_val_T2], f)
		else:
			save_file_name = os.path.split(t1_file_path)[1]
			save_file_name = os.path.splitext(save_file_name)[0]
			save_file_name = save_file_name.replace('T1', 't1').replace('t1', 't1_t2')
			print ("not saving ", save_file_name)
			with open('SubjLists_notsaved_EBDS_IBIS_resampled_1year.txt', 'a') as f:
				f.write('%s\n' % (save_file_name))

file_path_list = []
test_path_list = []

name_list = [f for f in listdir(path_data) if os.path.isdir(os.path.join(path_data, f))  ]
name_list.sort()
name_num = len(name_list)

for i in range(name_num):
	year_list = [f for f in listdir(os.path.join(path_data, name_list[i])) if os.path.isdir(os.path.join(path_data, name_list[i], f))  ]
	for j in range(len(year_list)):
		if year_list[j] == '1year':
			if os.path.isdir(os.path.join(path_data, name_list[i], year_list[j], 'anat')):
				file_list = [f for f in listdir(os.path.join(path_data, name_list[i], year_list[j], 'anat')) if os.path.isfile(os.path.join(path_data, name_list[i], year_list[j], 'anat', f)) and f.endswith('nrrd') ]
				if len(file_list) == 2:
					filename = os.path.join(path_data, name_list[i], year_list[j], 'anat') + '/' + name_list[i] + '-' + year_list[j] + '-T1.nrrd'
					file_path_list.append(filename)

random.seed(23)
random.shuffle(file_path_list)

total_num = len(file_path_list)

train_end = int(total_num * 0.8)
val_end = int(total_num * 0.9)

train_path_list = file_path_list[0: train_end]

val_path_list = file_path_list[train_end: val_end]

test_path_list = file_path_list[val_end:]

#################################################################

for _mode in ['train', 'val', 'test']:
	new_path = os.path.join(path_save, _mode)
	if not os.path.isdir(new_path):
		os.mkdir(new_path)

save_pickle(test_path_list, os.path.join(path_save, 'test'))
save_pickle(val_path_list, os.path.join(path_save, 'val'))
save_pickle(train_path_list, os.path.join(path_save, 'train'))