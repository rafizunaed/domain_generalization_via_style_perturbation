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
import numpy as np

from src.trainers.trainer_srm_il import ModelTrainer_IL
from src.trainers.trainer_srm_il_fl import ModelTrainer_SRM_IL_FL
from src.trainers.trainer_srm_il_fl_cons import ModelTrainer_SRM_IL_FL_CONS
from src.trainers.trainer_srm_il_cons import ModelTrainer_SRM_IL_CONS
from src.trainers.trainer_callbacks import set_random_state, AverageMeter, PrintMeter
from src.datasets.data import ThoracicDataset, get_train_transforms, get_valid_transforms, ThoracicDatasetDual,\
    ThoracicDatasetTest, collate_fn_img_level_ds
from src.models.build_model import create_model
from src.configs.training_configs import all_configs, configs_to_train
from src.utils.misc import remove_key_by_keyword_from_state_dict

def get_args():
    """
    get command line args
    """
    parser = ArgumentParser(description='train')
    parser.add_argument('--run_configs_list', type=str, nargs="*", default='prop_configs_list')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--image_resize_dim', type=int, default=256)
    parser.add_argument('--image_crop_dim', type=int, default=224)
    parser.add_argument('--do_grad_accum', type=bool, default=True)
    parser.add_argument('--grad_accum_step', type=int, default=4)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--use_focal_loss', type=bool, default=True)
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25)
    parser.add_argument('--focal_loss_gamma', type=float, default=2)
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    """
    main function
    """

    args = get_args()

    # print(1)

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)

    # check if there are duplicate weight saving paths
    unique_paths = np.unique([ x[1]['weight_saving_path'] for x in all_configs.items() ])
    assert len(all_configs.keys()) == len(unique_paths)

    run_configs_list = configs_to_train['prop_configs_list']

    for config_name in run_configs_list:
        configs = all_configs[config_name]
        
        dataset_root_dir = configs['dataset_root_dir']
        split_info_dict_dir = configs['split_info_dict_dir']
        split_dict = np.load(split_info_dict_dir, allow_pickle=True).item()
        
        for fold_number in range(args.n_folds):
            set_random_state(args.seed)
            # if fold_number <= 0:
                # continue
            print(f'Running fold-{fold_number} ....')
        
            train_fpaths = split_dict[f'fold_{fold_number}_train_fpaths']
            train_labels = split_dict[f'fold_{fold_number}_train_labels']
            val_fpaths = split_dict[f'fold_{fold_number}_val_fpaths']
            val_labels = split_dict[f'fold_{fold_number}_val_labels']
        
            if configs['method'] in ['srm_il', 'srm_fl', 'srm_il_fl']:
                train_dataset = ThoracicDataset(
                                    datasets_root_dir=dataset_root_dir,
                                    fpaths=train_fpaths,
                                    labels=train_labels,
                                    transform=get_train_transforms(args.image_resize_dim, args.image_crop_dim),
                                    use_SRM_IL=configs['use_srm_il'], 
                                    srm_il_min_value=configs['srm_il_min_value'], 
                                    srm_il_max_value=configs['srm_il_max_value'],
                                    )
            elif configs['method'] in ['srm_il_cons', 'srm_il_fl_cons', 'srm_il_cons']:
                train_dataset = ThoracicDatasetDual(
                                    datasets_root_dir=dataset_root_dir,
                                    fpaths=train_fpaths,
                                    labels=train_labels,
                                    transform=get_train_transforms(args.image_resize_dim, args.image_crop_dim),
                                    srm_il_min_value=configs['srm_il_min_value'], 
                                    srm_il_max_value=configs['srm_il_max_value'],
                                    )
                
            val_dataset = ThoracicDatasetTest(
                                datasets_root_dir=dataset_root_dir,
                                fpaths=val_fpaths,
                                labels=val_labels,
                                transform=get_valid_transforms(args.image_resize_dim, args.image_crop_dim),
                                )
            train_loader = DataLoader(
                                train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.n_workers,
                                drop_last=True,
                                collate_fn=collate_fn_img_level_ds,
                                )
            val_loader = DataLoader(
                                val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_workers,
                                drop_last=False,
                                collate_fn=collate_fn_img_level_ds,
                                )
    
            model = create_model(configs['backbone_architecture'], args.num_classes, init_srm_fl=configs['init_srm_fl'], randomization_stage=configs['randomization_stage'])
            if configs['checkpoint_root_path'] is not None:
                print('loading checkpoint!')
                wpath = configs['checkpoint_root_path']
                checkpoint = torch.load(f'{wpath}/fold{fold_number}/checkpoint_best_auc_fold{fold_number}.pth')
                print('fold {} loss score: {:.4f}'.format(fold_number, checkpoint['val_loss']))
                print('fold {} auc score: {:.4f}'.format(fold_number, checkpoint['val_auc']))
                model.load_state_dict(remove_key_by_keyword_from_state_dict(checkpoint['Model_state_dict']), strict=False)
                del checkpoint
            else:
                print('no checkpoint path is given!')
                    
            if configs['method'] in ['srm_il']:
                trainer_args = {
                        'model': model,
                        'Loaders': [train_loader, val_loader],
                        'metrics': {
                            'loss': AverageMeter,
                            'auc': PrintMeter,
                            },
                        'checkpoint_saving_path': configs['weight_saving_path'],
                        'lr': args.lr,
                        'epochsTorun': configs['epochs'],
                        'gpu_ids': args.gpu_ids,
                        'do_grad_accum': args.do_grad_accum,
                        'grad_accum_step': args.grad_accum_step,
                        'use_ema': args.use_ema,
                        'use_focal_loss': args.use_focal_loss,
                        'focal_loss_alpha': args.focal_loss_alpha,
                        'focal_loss_gamma': args.focal_loss_gamma,
                        'fold': fold_number,
                        }      
                trainer = ModelTrainer_IL(**trainer_args)
                trainer.fit()
                
            elif configs['method'] in ['srm_fl', 'srm_il_fl']:
                trainer_args = {
                        'model': model,
                        'Loaders': [train_loader, val_loader],
                        'metrics': {
                            'loss': AverageMeter,
                            'cls_loss': AverageMeter,
                            'content_loss': AverageMeter,
                            'style_loss': AverageMeter,
                            'auc': PrintMeter,
                            },
                        'checkpoint_saving_path': configs['weight_saving_path'],
                        'lr': args.lr,
                        'epochsTorun': configs['epochs'],
                        'gpu_ids': args.gpu_ids,
                        'do_grad_accum': args.do_grad_accum,
                        'grad_accum_step': args.grad_accum_step,
                        'use_ema': args.use_ema,
                        'use_focal_loss': args.use_focal_loss,
                        'focal_loss_alpha': args.focal_loss_alpha,
                        'focal_loss_gamma': args.focal_loss_gamma,
                        'fold': fold_number,
                        'eta': configs['eta'],
                        }
        
                trainer = ModelTrainer_SRM_IL_FL(**trainer_args)
                trainer.fit()
            
            elif configs['method'] in ['srm_il_fl_cons']:
                trainer_args = {
                        'model': model,
                        'Loaders': [train_loader, val_loader],
                        'metrics': {
                            'loss': AverageMeter,
                            'cls_loss': AverageMeter,
                            'content_loss': AverageMeter,
                            'style_loss': AverageMeter,
                            'auc': PrintMeter,
                            'ccr_loss': AverageMeter,
                            'pdr_loss': AverageMeter,
                            },
                        'checkpoint_saving_path': configs['weight_saving_path'],
                        'lr': args.lr,
                        'epochsTorun': configs['epochs'],
                        'gpu_ids': args.gpu_ids,
                        'do_grad_accum': args.do_grad_accum,
                        'grad_accum_step': args.grad_accum_step,
                        'use_ema': args.use_ema,
                        'use_focal_loss': args.use_focal_loss,
                        'focal_loss_alpha': args.focal_loss_alpha,
                        'focal_loss_gamma': args.focal_loss_gamma,
                        'fold': fold_number,
                        'eta': configs['eta'],
                        }
        
                trainer = ModelTrainer_SRM_IL_FL_CONS(**trainer_args)
                trainer.fit()
                
            elif configs['method'] in ['srm_il_cons']:
                trainer_args = {
                        'model': model,
                        'Loaders': [train_loader, val_loader],
                        'metrics': {
                            'loss': AverageMeter,
                            'cls_loss': AverageMeter,
                            'ccr_loss': AverageMeter,
                            'pdr_loss': AverageMeter,
                            'auc': PrintMeter,
                            },
                        'checkpoint_saving_path': configs['weight_saving_path'],
                        'lr': args.lr,
                        'epochsTorun': configs['epochs'],
                        'gpu_ids': args.gpu_ids,
                        'do_grad_accum': args.do_grad_accum,
                        'grad_accum_step': args.grad_accum_step,
                        'use_ema': args.use_ema,
                        'use_focal_loss': args.use_focal_loss,
                        'focal_loss_alpha': args.focal_loss_alpha,
                        'focal_loss_gamma': args.focal_loss_gamma,
                        'fold': fold_number,
                        }
        
                trainer = ModelTrainer_SRM_IL_CONS(**trainer_args)
                trainer.fit()
        
if __name__ == '__main__':
    main()