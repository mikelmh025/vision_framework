import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

# Custom imports
import sys
sys.path.append('models')
import model_factory
import loss_factory 
import optimizer_factory 
import train_loops.prior_train as prior_train
from train_loops.train_control import train_control
import test
import datasets
import utils    


def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError('Error loading config file.')
    return config



def model2device(full_package, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        full_package['model'] = nn.DataParallel(full_package['model'])
        config['train']['batch_size'] = config['train']['batch_size']*torch.cuda.device_count()
        config['test']['batch_size'] = config['test']['batch_size']*torch.cuda.device_count()
    full_package['model'] = full_package['model'].to(device)
    full_package['device'] = device
    return full_package, config

def train_eval_loop(full_package, config):
    # Training loop
    print('==> Start training..')
    full_package['it_global'] = 0
    full_package['epoch'] = 0

    while full_package['epoch'] < config['train']['epochs'] or full_package['it_global'] < config['train']['global_iteration']:
        # train loop
        train_loss = train_control(full_package, config['train']['loop_type'])
        # test loop
        acc = test.test_control(full_package, config['test']['loop_type'])
        full_package['scheduler'].step()
        print(f"Epoch {full_package['epoch']+ 1}, test Accuracy: {acc:.4f}")

        if config['train']['debug'] and full_package['epoch'] >= 5: break
    
    print('Finished Training')

def main(config):
    full_package = {'config': config}
    
    # Define the dataset and dataloader.
    # TODO different types of loaders: noise loader
    datasets.get_loaders(config,full_package)

    prior_train.prior_train_control(full_package)

    # Define the model, loss function, and optimizer.
    # TODO: Deal with loading pretrained models
    model_factory.get_model(config['model']['name'], pretrained=True, num_classes=config['data']['num_classes'],full_package=full_package)
    full_package, config = model2device(full_package, config)

    # TODO: optmizer lr boundaries
    # TODO: [Noisy] more loss functions
    loss_factory.get_loss_function(config['train']['loss_type'],full_package)
    optimizer_factory.get_optimizer(full_package)
    # TODO: deal with scheduler boundaries
    optimizer_factory.get_scheduler(full_package)

    train_eval_loop(full_package, config)

    prior_train.post_train(full_package)
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    

    config = utils.start_program(args)    
    
    np.random.RandomState(seed=config['general']['np_seed'])
    torch.manual_seed(config['general']['torch_seed'])


    main(config)
