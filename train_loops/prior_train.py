import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter

def prior_train_control(train_package):
    config = train_package['config']

    train_summary_writer = SummaryWriter(os.path.join(config['general']['save_root'], 'summaries/train'))
    eval_summary_writer = SummaryWriter(os.path.join(config['general']['save_root'], 'summaries/eval'))
    fp_log_res = open(os.path.join(config['general']['save_root'], 'results_log.txt'), 'w')

    train_package['summary_writers'] = {'train':train_summary_writer, 'eval':eval_summary_writer, 'fp_log_res':fp_log_res}

    if train_package['config']['train']['loop_type'] == 'drops':
        return prior_drops(train_package)
    else:
        return 

def prior_drops(train_package):
    config = train_package['config']
    imbalance_info = train_package['imbalance_info']

    # TODO: optimize this for speed
    # TODO: Don't hardcode this
    # revise class-level weights
    if config['train']['drops']['re_weight_type'] == 'none' or config['train']['drops']['is_upsampling']:
        if config['data']['dataset_name'] == 'cifar10':
            imbalance_info['samples_per_cls'] = [5000] * config['data']['num_classes']
        elif config['data']['dataset_name'] == 'cifar100':
            imbalance_info['samples_per_cls'] = [500] * config['data']['num_classes']
            
    if config['train']['drops']['re_weight_type'] == 'sqrt':
        raise NotImplementedError
        imbalance_info['samples_per_cls'] = tf.sqrt(tf.dtypes.cast(imbalance_info['samples_per_cls'], dtype=tf.float64))
        imbalance_info['samples_per_cls'] = imbalance_info['samples_per_cls'].numpy().tolist()

    # Initialize g_y to be the same as uniform | prior p(y)| 1/p(y)
    # Default is set to be 'uniform'
    if config['train']['drops']['metric_base'] == 'uniform':
        g_y = [1] * config['data']['num_classes']
    elif config['train']['drops']['metric_base'] == 'prior' and config['train']['drops']['re_weight_type'] == 'prior':
        g_y = [1/i for i in imbalance_info['samples_per_cls']]
    elif config['train']['drops']['metric_base'] == 'recip_prior' and config['train']['drops']['re_weight_type'] == 'prior':
        g_y = imbalance_info['samples_per_cls']
    g_y = g_y / np.sum(g_y)
    alpha_y = [1/i for i in imbalance_info['samples_per_cls']]
    alpha_y = torch.tensor(alpha_y).float()
    alpha_y *= torch.sum(torch.tensor(imbalance_info['samples_per_cls']).float())


    lambd = 1.0
    # set the r_list to be the u in the constraint D(u, g) < delta
    r_list = g_y

    g_y = torch.tensor(g_y).float()
    r_list = torch.tensor(r_list).float()

    train_package['drops_info'] = {
        'g_y': g_y,
        'alpha_y': alpha_y,
        'lambd': lambd,
        'r_list': r_list,
    }
    # 'delta': config['train']['drops']['eps']

    return train_package


def post_train(train_package):
    train_summary_writer, eval_summary_writer, fp_log_res = \
        train_package['summary_writers']['train'], train_package['summary_writers']['eval'], train_package['summary_writers']['fp_log_res']
    
    fp_log_res.close()
    train_summary_writer.close()
    eval_summary_writer.close()

