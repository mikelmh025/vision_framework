import torch.optim as optim

def get_optimizer(full_package):
    config = full_package['config']
    optimizer_type, learning_rate = config['train']['optimizer_type'], config['train']['learning_rate']

    if optimizer_type == 'sgd':
        opt_ = optim.SGD(full_package['model'].parameters(), lr=learning_rate)
    elif optimizer_type == 'adam':
        opt_ = optim.Adam(full_package['model'].parameters(), lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        opt_ = optim.RMSprop(full_package['model'].parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    full_package['optimizer'] = opt_


def get_scheduler(full_package):
    config = full_package['config']
    scheduler_type = config['train']['scheduler_type']

    if scheduler_type == 'step':
        scheduler_gamma,scheduler_step_size = config['train']['scheduler_gamma'],config['train']['scheduler_step_size']
        scheduler = optim.lr_scheduler.StepLR(full_package['optimizer'], step_size=scheduler_step_size, gamma=scheduler_gamma)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    full_package['scheduler'] = scheduler

