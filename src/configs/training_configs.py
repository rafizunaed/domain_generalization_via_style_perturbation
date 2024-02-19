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

from src.configs.base_configs import CHEXPERT_MIMIC_DATASETS_ROOT_DIR, CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH

configs_to_train = {
    'prop_configs_list': ['init_pretrain_on_md_chexpert_mimic', 'srm_il_md_chexpert_mimic', 'srm_il_fl_S1_md_chexpert_mimic',\
                     'srm_il_fl_S2_md_chexpert_mimic', 'srm_il_fl_S2_cons_md_chexpert_mimic']
        }

configs_to_test = {
    'prop_model': 'srm_il_fl_S2_cons_md_chexpert_mimic',
    }
    
all_configs = {
    'init_pretrain_on_md_chexpert_mimic':{
        'weight_saving_path': './weights/init_pretrain_on_md_chexpert_mimic/',
        'checkpoint_root_path': None,
        'method': 'srm_il',
        'epochs': 15, 
        'use_srm_il': False,
        'srm_il_min_value': 0,
        'srm_il_max_value': 255,
        'init_srm_fl': False,
        'randomization_stage': None,
        'eta': 0,
        'split_info_dict_dir': CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH,
        'dataset_root_dir': CHEXPERT_MIMIC_DATASETS_ROOT_DIR,
        },
    
    'srm_il_md_chexpert_mimic':{
        'weight_saving_path': './weights/srm_il_md_chexpert_mimic_w_load/',
        'checkpoint_root_path': './weights/init_pretrain_on_md_chexpert_mimic/',
        'method': 'srm_il',
        'epochs': 5, 
        'use_srm_il': True,
        'srm_il_min_value': 0,
        'srm_il_max_value': 255,
        'init_srm_fl': False,
        'randomization_stage': None,
        'eta': 0,
        'split_info_dict_dir': CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH,
        'dataset_root_dir': CHEXPERT_MIMIC_DATASETS_ROOT_DIR,
        },
    
    'srm_il_fl_S1_md_chexpert_mimic':{
        'weight_saving_path': './weights/srm_il_fl_S1_md_chexpert_mimic/',
        'checkpoint_root_path': './weights/srm_il_md_chexpert_mimic/',
        'method': 'srm_il_fl',
        'epochs': 5, 
        'use_srm_il': True,
        'srm_il_min_value': 0,
        'srm_il_max_value': 255,
        'init_srm_fl': True,
        'randomization_stage': 'S1',
        'eta': 0.01,
        'split_info_dict_dir': CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH,
        'dataset_root_dir': CHEXPERT_MIMIC_DATASETS_ROOT_DIR,
        },
    
    'srm_il_fl_S2_md_chexpert_mimic':{
        'weight_saving_path': './weights/srm_il_fl_S2_md_chexpert_mimic/',
        'checkpoint_root_path': './weights/srm_il_fl_S1_md_chexpert_mimic/',
        'method': 'srm_il_fl',
        'epochs': 5, 
        'use_srm_il': True,
        'srm_il_min_value': 0,
        'srm_il_max_value': 255,
        'init_srm_fl': True,
        'randomization_stage': 'S2',
        'eta': 0.01,
        'split_info_dict_dir': CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH,
        'dataset_root_dir': CHEXPERT_MIMIC_DATASETS_ROOT_DIR,
        },
    
    'srm_il_fl_S2_cons_md_chexpert_mimic':{
        'weight_saving_path': './weights/srm_il_fl_S2_cons_md_chexpert_mimic/',
        'checkpoint_root_path': './weights/srm_il_fl_S2_md_chexpert_mimic/',
        'method': 'srm_il_fl_cons',
        'epochs': 20, 
        'use_srm_il': True,
        'srm_il_min_value': 0,
        'srm_il_max_value': 255,
        'init_srm_fl': True,
        'randomization_stage': 'S2',
        'eta': 0.01,
        'split_info_dict_dir': CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH,
        'dataset_root_dir': CHEXPERT_MIMIC_DATASETS_ROOT_DIR,
        },
    
    'srm_il_cons_md_chexpert_mimic':{
        'weight_saving_path': './weights/srm_il_cons_md_chexpert_mimic/',
        'checkpoint_root_path': './weights/srm_il_md_chexpert_mimic/',
        'method': 'srm_il_cons',
        'epochs': 20, 
        'use_srm_il': True,
        'srm_il_min_value': 0,
        'srm_il_max_value': 255,
        'init_srm_fl': False,
        'randomization_stage': None,
        'eta': 0,
        'split_info_dict_dir': CHEXPERT_MIMIC_SPLIT_INFO_DICT_PATH,
        'dataset_root_dir': CHEXPERT_MIMIC_DATASETS_ROOT_DIR,
        },
    }