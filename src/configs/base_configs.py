# MIT License
#
# Copyright (c) 2024 Mohammad Zunaed, mHealth Lab, BUET
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

DEVICE = torch.device("cuda:0")

CHEXPERT_MIMIC_DATASETS_ROOT_DIR = '/mnt/D4B4406BB44051E2/ML/domain_agnostic_codebase/datasets/'
BRAX_DATASET_ROOT_DIR = '/mnt/D4B4406BB44051E2/ML/datasets/BRAX_CXR/BRAX_CXR_512/'
NIH_CXR14_DATASET_ROOT_DIR = '/mnt/D4B4406BB44051E2/ML/datasets/NIH_CXR/NIH_CXR_256/'
VinDr_CXR_DATASET_ROOT_DIR = '/mnt/D4B4406BB44051E2/ML/datasets/VinDr_CXR/VinDr_CXR_256/'

CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH = '/mnt/D4B4406BB44051E2/ML/domain_agnostic_codebase/datasets/split_and_test_dicts/chexpert_mimic_split_info_dict.npy'
BRAX_TEST_INFO_DICT_PATH = '/mnt/D4B4406BB44051E2/ML/datasets/BRAX_CXR/BRAX_labels/brax_split_info_dict_only_frontal_uones_5f_train_val_holdout_test_NOV23.npy'
NIH_CXR14_TEST_INFO_DICT_PATH = '/mnt/D4B4406BB44051E2/ML/datasets/NIH_CXR/NIH_CXR_labels/NIH_img_level_split_info_dict_7cls_NOV23_split_ratio_rs1_4690_rs2_1234.npy'
VinDr_CXR_TEST_INFO_DICT_PATH = '/mnt/D4B4406BB44051E2/ML/datasets/VinDr_CXR/VinDr_CXR_labels/VinDr_10cls_split_info_dict_5f_train_val_holdout_test_DEC23.npy'