import yaml
import os
import torch
import uuid
import time
import numpy as np



def merge_configs(base_config, override_config):
    for key, value in override_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            # Recursively merge nested dictionaries
            merge_configs(base_config[key], value)
        else:
            # Update or add the key-value pair
            base_config[key] = value

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    inherit_from = config.get('inherit_from')
    if inherit_from:
        # Load values from the inherited configuration file
        inherited_config = load_config(inherit_from)
        merge_configs(inherited_config, config)
        config = inherited_config

    return config

def parse_keys(keys, dictionary):
    return [dictionary[key] for key in keys]

def print_config(config_dict, indent=0):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            print_config(value, indent + 4)
        else:
            print(' ' * indent + f"{key}: {value}")


        

def update_save_dir(config):    
    save_root = config['general']['save_root']
    dataset = config['data']['dataset_name']
    loss = config['train']['loss_type']
    run_id = config['general']['run_id']
    model_name = config['model']['name']

    if config['train']['loop_type'] =='drops':
        imb_ratio = config['train']['imbalance_factor']
        save_root = os.path.join(save_root, f'{dataset}_{imb_ratio}')
        save_root = os.path.join(save_root, f'_loss_{loss}_{run_id}')
        # save_root = os.path.join(save_root, f'{model_name}')
        config['general']['save_root'] = save_root
        config['general']['save_model_dir'] = os.path.join(save_root, 'model')
        
    else:
        # config['general']['save_root'] = os.path.join(save_root, model_name)
        imb_ratio = config['train']['imbalance_factor']
        save_root = os.path.join(save_root, f'{dataset}_{imb_ratio}')
        save_root = os.path.join(save_root, f'_loss_{loss}_{run_id}')
        config['general']['save_root'] = save_root
        config['general']['save_model_dir'] = os.path.join(save_root, 'model')

    
def create_save_dir(config):
    # Create save dir
    print('==> Saving to:', config['general']['save_root'])
    if not os.path.exists(config['general']['save_root']):
        os.makedirs(config['general']['save_root'])
    
    

def mange_cuda_devices(config):
    if not os.path.exists(config['general']['save_model_dir']):
        os.makedirs(config['general']['save_model_dir'])
    if config['general']['CUDA_VISIBLE_DEVICES'] != '-1':
        print('CUDA_VISIBLE_DEVICES: {}'.format(config['general']['CUDA_VISIBLE_DEVICES']))
        os.environ["CUDA_VISIBLE_DEVICES"] = config['general']['CUDA_VISIBLE_DEVICES']

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [f"{torch.cuda.get_device_name(i)} (GPU {i})" for i in range(device_count)]
        print(f"==> Using GPU: {', '.join(devices)}")
    else:
        print("==> Using CPU")

# Load config, print config, modify save dir, create dir, return config
def start_program(args):
    timestamp = str(int(time.time()))#[-6:]
    run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"

    print('==> Loading config:', args.config)
    config = load_config(args.config)
    config['general']['run_id'] = run_id

    update_save_dir(config)

    print('==> Using config:')
    print_config(config)

    # Create save dir (mkdir)
    create_save_dir(config)

    # set CUDA_VISIBLE_DEVICES, Show CUDA devices,
    mange_cuda_devices(config)
        

    # Save config
    config_file_path = os.path.join(config['general']['save_root'], f'{run_id}.yaml')
    print('==> Saving run config to:', config_file_path)
    with open(config_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return config

def save_train_package(train_package, file_name,keys_to_save=None):
    if keys_to_save is None:
        keys_to_save = ['config','model','optimizer','epoch']

    save_dict = {}
    for key in keys_to_save:
        save_dict[key] = train_package[key]
    torch.save(save_dict, file_name)

def load_train_package(file_name,keys_to_load=None,train_package=None):
    if keys_to_load is None:
        keys_to_load = ['config','model','optimizer','epoch']

    train_package_loaded = torch.load(file_name)
    for key in keys_to_load:
        train_package[key] = train_package_loaded[key]
    return train_package

def tensor_to_string(tensor, precision=4):
    np.set_printoptions(precision=precision, floatmode='fixed', suppress=True)
    return np.array2string(tensor.numpy())


# was from get_save_dir: Since we are using uuid to generate run_id, we don't need to check lock file
#
#     # If lock file exists, exit
# lock_files = [file for file in os.listdir(config['general']['save_root']) if file.endswith('.lock')]
# if len(lock_files) > 0:
#     print('==> Lock file exists. Exiting.')
#     exit()

# # Create lock file
# if config['general']['train_lock'] == True:
#     lock_file_path = os.path.join(config['general']['save_root'], f'{run_id}.lock')
#     print('==> Creating lock file:', lock_file_path)
#     open(lock_file_path, 'w').close()


# was from get_save_dir: Since we are using uuid to generate run_id, we don't need to Whip existing files
#
# Whip existing files
# if config['general']['whip_existing_files']:
#     print('==> Whipping existing files in dir: ', config['general']['save_root'], 'whip_existing_files: True !')
#     for file in os.listdir(config['general']['save_root']):
#         os.remove(os.path.join(config['general']['save_root'], file))