from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, RandomSampler  
from torch.utils.data import random_split
import copy


import numpy as np
import random

def get_loaders(config, full_package):
    print('==> Preparing data..')
    train_transform, test_transform = get_transform(config)

    # Continue: 6/20/2023 Afternoon
    trainset, testset, valset = get_dataset(config,train_transform,test_transform)
        
    # Imbalance loader. config['train']['imbalance_factor'] = 1 means no imbalance
    full_package = get_imbalance_loader(config, full_package,trainset)
    

    trainloader = DataLoader(trainset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'])
    valloader = DataLoader(valset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['test']['num_workers'])
    testloader = DataLoader(testset, batch_size=config['test']['batch_size'], shuffle=False, num_workers=config['test']['num_workers'])

    full_package['trainloader'],full_package['valloader'],full_package['testloader'] = trainloader,valloader,testloader

    if config['train']['loop_type'] == 'peer':
        peer_sampler_x,peer_sampler_y = RandomSampler(trainset, replacement=True),RandomSampler(trainset, replacement=True)
        full_package['peerloader_x'] = DataLoader(trainset, batch_size=config['train']['batch_size'], \
                                                  shuffle=False, sampler=peer_sampler_x, num_workers=config['train']['num_workers'])
        full_package['peerloader_y'] = DataLoader(trainset, batch_size=config['train']['batch_size'], \
                                                  shuffle=False, sampler=peer_sampler_y, num_workers=config['train']['num_workers'])


    return full_package

def split_train_val(config, trainset):

    # Split valset from trainset. 10% of the trainset or 5000 samples
    split_train_size = config['train']['train_size']*len(trainset) if config['train']['train_size'] < 1 else config['train']['train_size']
    val_size = len(trainset) - split_train_size
    split_a, split_b = random_split(range(len(trainset)), [int(split_train_size), int(val_size)])
    split_train_idx, val_idx = split_a.indices, split_b.indices

    split_trainset, valset = copy.deepcopy(trainset), copy.deepcopy(trainset)

    split_trainset.data = split_trainset.data[split_train_idx]
    split_trainset.targets = np.array(split_trainset.targets)[split_train_idx]

    valset.data = valset.data[val_idx]
    valset.targets = np.array(valset.targets)[val_idx]

    if not config['train']['train_on_full']: trainset = split_trainset
    if val_size > 20000: print('Warning: val_size is larger than 20000')

    return trainset, valset

# data_set is subset of trainset
def get_imbalance_loader(config, full_package,data_set):
    y_clean = np.array(data_set.targets)
    samples_per_cls = get_cls_num(y_clean, config['train']['imbalance_factor'], 
                                    cls_num=config['data']['num_classes'])
    selected_idx = get_cls_idx(num_per_cls=samples_per_cls, y_train=y_clean,
                            is_upsampling=config['train']['imbalance_is_upsampling'])
    
    data_set.data = data_set.data[selected_idx] # Update data / select only the selected indices
    data_set.targets = np.array(data_set.targets)[selected_idx] # Update targets / select only the selected indices
    full_package['imbalance_info'] = {'samples_per_cls': samples_per_cls, 'selected_idx': selected_idx}

    return full_package

# def get_dataset(name, root, train=True, transform=None, download=True):
def get_dataset(config, train_transform, test_transform, download=True):
    name = config['data']['dataset_name']
    train_root, test_root = config['data']['train_dir'], config['data']['test_dir']

    if name.lower() == 'cifar10':
        trainset = CIFAR10(root=train_root, train=True, transform=train_transform, download=download)
        testset  = CIFAR10(root=test_root, train=False, transform=test_transform, download=download)
    elif name.lower() == 'cifar100':
        trainset = CIFAR100(root=train_root, train=True, transform=train_transform, download=download)
        testset  = CIFAR100(root=test_root, train=False, transform=test_transform, download=download)
    elif name.lower() == 'mnist':
        trainset = MNIST(root=train_root, train=True, transform=train_transform, download=download)
        testset  = MNIST(root=test_root, train=False, transform=test_transform, download=download)
    elif name.lower() == 'clothing1mpp':
        from loaders.clothing1mpp import Clothing1mPP

        path_dict ={
            'root': config['data']['root'], # Root dir of the dataset that includes folders of images: Jacket, Shirt, etc.
            'full_label_json_path': config['data']['full_label_json_path'], # Json file that includes all the labels
            'split_path':config['data']['split_path'] # Dir includes to test_ids.pt, train_ids.pt, val_ids.pt
        }
        image_size = config['data']['image_size']
        label_types = config['data']['label_types']
        debug = config['train']['debug']
        
        # Load datasets
        trainset = Clothing1mPP(path_dict, image_size,debug=debug,split='train',label_types=label_types,transform=train_transform)
        pre_load = trainset.pre_load
        valset = Clothing1mPP(path_dict, image_size,debug=debug,split='val',label_types=label_types,pre_load=pre_load,transform=test_transform)
        testset = Clothing1mPP(path_dict, image_size,debug=debug,split='test',label_types=label_types,pre_load=pre_load,transform=test_transform)
        return trainset, testset, valset

    elif name.lower() == 'custom':
        # Assuming your images are in a folder structure like:
        # root/dog/xxx.png
        # root/cat/yyy.png
        # ...
        # Where 'root' is specified in your yaml file.
        return ImageFolder(root=root, transform=transform)
    else:
        raise ValueError(f'Dataset {name} not recognized. Choose from ["cifar10", "cifar100", "mnist"]')
    
    trainset, valset = split_train_val(config, trainset)

    return trainset, testset, valset

def get_transform(config):
    name = config['data']['dataset_name']
    image_size = config['data']['image_size']

    if name.lower() == 'cifar10':
        if image_size!=32: print('Warning: cifar10 image_size is not 32')
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif name.lower() == 'cifar100':
        if image_size!=32: print('Warning: cifar10 image_size is not 32')
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize the image to the desired crop size
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize the image to the desired crop size
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
        ])
        

    
    return train_transform, test_transform

def get_cls_num(y_train, imb_factor=None, cls_num=10):
    """Get a list of image numbers for each class.
    Given cifar version Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-imb * 0);
    img min: 5000 / 500 * e^(-imb * int(dataset_name - 1))
    exp(-imb * (int(dataset_name) - 1)) = img_max / img_min
    Args:
        dataset_name: str, 'cifar10', 'cifar100
        y_train: the training label
        imb_factor: float, imbalance factor: img_min/img_max, None if geting
        default cifar data number
    Returns:
        img_num_per_cls: a list of number of images per class
    """
    img_max = len(list(np.where(np.array(y_train) == 0)[0]))
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        idx_this_class = list(np.where(np.array(y_train) == cls_idx)[0])
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(min(int(num), len(idx_this_class)))
    return img_num_per_cls

def get_cls_idx(num_per_cls, y_train, is_upsampling=False):
    """Get the seleted index for the training images.
    Given number of selected images per class, return the image indexes adopted.
    Args:
        num_per_cls: a list of number of images per class
        y_train: the clean label for class selection
        is_upsampling: whether up sampling to strick a balance
    Returns:
        selected_idx: a list of selected images for training use
    """
    cls_num = len(num_per_cls)
    selected_idx = []
    for cls_idx in range(cls_num):
        idx_this_class = list(np.where(np.array(y_train) == cls_idx)[0])
        indices = random.sample(idx_this_class, num_per_cls[cls_idx])
        if is_upsampling:
            up_samling = random.choices(indices, k=int(50000/cls_num)-len(indices))
            selected_idx.extend(up_samling)
        selected_idx.extend(indices)
    return selected_idx