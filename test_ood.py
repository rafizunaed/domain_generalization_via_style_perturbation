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

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch import nn
import time

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.datasets.data import ThoracicDatasetTest, get_valid_transforms, collate_fn_img_level_ds
from src.trainers.trainer_callbacks import set_random_state
from src.models.build_model import create_model
from src.configs.base_configs import DEVICE, BRAX_DATASET_ROOT_DIR, NIH_CXR14_DATASET_ROOT_DIR, VinDr_CXR_DATASET_ROOT_DIR,\
    BRAX_TEST_INFO_DICT_PATH, NIH_CXR14_TEST_INFO_DICT_PATH, VinDr_CXR_TEST_INFO_DICT_PATH 
from src.configs.training_configs import all_configs, configs_to_test

def get_args():
    parser = ArgumentParser(description='test')
    parser.add_argument('--run_config', type=str, default='prop_model')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_resize_dim', type=int, default=256)
    parser.add_argument('--image_crop_dim', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--test_dataset', type=str, default='brax')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)
            
    # get configs
    run_config = configs_to_test[args.run_config] 
    configs = all_configs[run_config]
    weight_saving_path = configs['weight_saving_path']
    
    # dataset specific info
    assert args.test_dataset in ['brax', 'nih', 'vindr']
    if args.test_dataset == 'brax':
        dataset_root_dir = BRAX_DATASET_ROOT_DIR
        test_info_dict_path = BRAX_TEST_INFO_DICT_PATH
    elif args.test_dataset == 'nih':
        dataset_root_dir = NIH_CXR14_DATASET_ROOT_DIR
        test_info_dict_path = NIH_CXR14_TEST_INFO_DICT_PATH
    elif args.test_dataset == 'vindr':
        dataset_root_dir = VinDr_CXR_DATASET_ROOT_DIR
        test_info_dict_path = VinDr_CXR_TEST_INFO_DICT_PATH
    
    # get dataloader
    test_dict = np.load(test_info_dict_path, allow_pickle=True).item()
    test_fnames=test_dict['test_fpaths']
    test_labels=test_dict['test_labels']
    test_dataset = ThoracicDatasetTest(
                        datasets_root_dir=dataset_root_dir,
                        fpaths=test_fnames,
                        labels=test_labels,
                        transform=get_valid_transforms(args.image_resize_dim, args.image_crop_dim),
                        )              
    test_loader = DataLoader(
                        test_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=args.n_workers,
                        drop_last=False,
                        collate_fn=collate_fn_img_level_ds,
                        )   
    
    all_mean_auc = []
    all_clswise_auc = []
    
    for fold_number in range(args.n_folds):
        set_random_state(args.seed)
        # if fold_number != 0:
            # continue
        print('Running fold-{} ....'.format(fold_number))
           
        all_targets = []
        all_probabilities = []
        
        model = create_model(configs['backbone_architecture'], args.num_classes, init_srm_fl=configs['init_srm_fl'], randomization_stage=configs['randomization_stage'])
        checkpoint = torch.load(weight_saving_path+f'/fold{fold_number}/checkpoint_best_auc_fold{fold_number}.pth')
        print('fold {} loss score: {:.4f}'.format(fold_number, checkpoint['val_loss']))
        print('fold {} auc score: {:.4f}'.format(fold_number, checkpoint['val_auc']))
        model.load_state_dict(checkpoint['Model_state_dict'])
        model = model.to(DEVICE)
        if len(args.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=args.gpu_ids)
        model.eval()                  
        del checkpoint
        
        torch.set_grad_enabled(False)
        with torch.no_grad():
            for itera_no, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                images = data['image'].to(DEVICE) 
                targets = data['target'].to(DEVICE)                  
                
                with torch.cuda.amp.autocast():
                    out = model(images)
                    
                all_targets.append(targets.cpu().data.numpy())
                y_prob = out['logits'].cpu().detach().clone().float().sigmoid()
                
                if args.test_dataset == 'brax':
                    pass
                elif args.test_dataset == 'nih':
                    y_prob = torch.cat([y_prob[:,0:1], y_prob[:,1:2], y_prob[:,3:4], y_prob[:,4:5], y_prob[:,5:6], y_prob[:,6:7], y_prob[:,7:8]], dim=1)
                elif args.test_dataset == 'vindr':
                    y_prob = torch.cat([y_prob[:,0:1], y_prob[:,1:2], y_prob[:,3:4], y_prob[:,4:5], y_prob[:,10:11],\
                                        y_prob[:,7:8], y_prob[:,5:6], y_prob[:,6:7], y_prob[:,11:12], y_prob[:,13:14]], dim=1)
                    
                all_probabilities.append(y_prob.numpy())
            
        all_targets = np.concatenate(all_targets)
        all_probabilities = np.concatenate(all_probabilities)
        
        auc = roc_auc_score(all_targets, all_probabilities)
        all_mean_auc.append(auc)
        print(f'fold{fold_number} auc score: {auc*100}')
        time.sleep(1)       
        all_clswise_auc.append(roc_auc_score(all_targets, all_probabilities, average=None))
     
    all_mean_auc = np.array(all_mean_auc)
    all_clswise_auc = np.stack(all_clswise_auc)  
    print('5-fold auc mean: {:.2f}, std: {:.2f}'.format(all_mean_auc.mean(0)*100, all_mean_auc.std(0)*100))
    all_clswise_auc_mean = np.array([float('{:.2f}'.format(x*100)) for x in all_clswise_auc.mean(0)])
    all_clswise_auc_std = np.array([float('{:.2f}'.format(x*100)) for x in all_clswise_auc.std(0)])
    print(all_clswise_auc_mean)
    print(all_clswise_auc_std)
    
if __name__ == '__main__':
    main()