import argparse
import os
from solver import Solver
from torch.backends import cudnn
import random
from data_loader import create_train_dataloader,create_test_dataloader
def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','ASM_U_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = 0.0001
    epoch =300
    decay_epoch =[100,150,200]
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch
    config.dataset_root='/home/wangjian/CrowdCounting/WorldExpo/'
    config.batch_size=1

    print(config)
        
    train_dataloader = create_train_dataloader(config.dataset_root, use_flip=True, batch_size=config.batch_size)
    test_dataloader  = create_test_dataloader(config.dataset_root)

    solver = Solver(config, train_dataloader, test_dataloader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=0)

    config = parser.parse_args()
    main(config)
