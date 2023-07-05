# import tensorflow as tf
# # tf.config.run_functions_eagerly(True)
# tf.executing_eagerly()

import os
import numpy as np
import json
from PIL import Image
import imghdr
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# class clothing1mppLoader:
class Clothing1mPP(Dataset):
    def __init__(self, path_dict, image_size,debug=False,label_types=['class'],
                 imb_factor=None, is_upsampling=False,split='train',pre_load=None,train_on_full=False,transform=None):
        print("#############################################")
        print("Loading Clothing1mPP dataset: ", split)
        
        self.debug = debug
        self.root = path_dict['root']
        self.label_json_path = path_dict['full_label_json_path']
        self.split_path = path_dict['split_path']

        self.label_types = label_types
        self.image_size = image_size
        self.split = split
        self.train_on_full = train_on_full

        # Imbalance factor for training, ignore if not using imbalance training 
        self.imb_factor = imb_factor 
        self.is_upsampling = is_upsampling

        if transform is None:
            if self.split == 'train': 
                self.transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),  # Resize the image to the desired crop size
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
                ])
            
        else:
            self.transform = transform

        # Load all the images and labels takes time. You can load once for training, and use pre_load for validation and testing
        if pre_load is None:
            print("Loading data from scratch")
            self.data, self.targets = self.load_clothing1mpp()
            self.pre_load = (self.data.copy(), self.targets.copy())
        else:
            print("Using pre-loaded data")
            self.data, self.targets = pre_load

        print("Splitting data")
        self.split_train_val_test()
        print(f"Done loading data, split: {split}. total images: {len(self.data)}")
        print("#############################################")

        self.data = np.array(self.data)
        self.targets = np.array(self.targets).astype(np.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.targets[idx]

        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure the image has RGB channels
            image = self.transform(img)
        assert image.shape == (3, self.image_size, self.image_size), f"image shape: {image.shape}"
        return image, label

    
    def split_train_val_test(self):
        if self.split == 'train':
            ids = torch.load(os.path.join(self.split_path, 'train_ids.pt'))
        elif self.split == 'val':
            ids = torch.load(os.path.join(self.split_path, 'val_ids.pt'))
        elif self.split == 'test':
            ids = torch.load(os.path.join(self.split_path, 'test_ids.pt'))

        # Debug mode: 
        # Filter the ids to make sure they are within the range of the debug dataset
        if self.debug:
            max_idx = len(self.data)
            ids = [idx for idx in ids if idx < max_idx]

        self.data = [self.data[idx] for idx in ids]
        self.targets = self.targets[ids]

        if self.split == 'train' and self.imb_factor is not None:
            assert self.label_types == ['Class'], 'Only support imbalance training on Class label'
            self.get_imb()

    # path_list: List of image paths
    # labels: np array of labels, shape: (num_images, num_labels) i.e. (1000, 4)
    def load_clothing1mpp(self):

        # Based on Meta data, convert the string label to integer label
        def get_label(class_label, meta_dict,attri_label=None,label_type='Class',attribute_threshold=10):
            if label_type=='Class':
                class_types = list(meta_dict.keys())
                label = class_types.index(class_label)
            else:
                class_types = meta_dict[class_label][label_type]
                label = class_types.index(attri_label)
            return label

        # Load the json file
        with open(self.label_json_path, 'r') as f:
            data_dict = json.load(f)
        label_list = data_dict['labels']
        meta_dict = data_dict['meta_data']


        # valid_subfix = ['jpg', 'png', 'gif', 'bmp', 'jpeg']
        path_list = []
        label_dict = {item: [] for item in self.label_types}

        for item in label_list:
            # if item['file_path'].split('.')[-1] not in valid_subfix: continue

            # Get path info
            path_list.append(self.root + item['file_path'])            
            
            # Get label info
            for key in self.label_types:
                if key == 'Class':
                    label_ = get_label(item['Labels'], meta_dict, label_type=key)
                else:
                    label_ = get_label(item['Labels'], meta_dict, label_type=key,attri_label=item['attributes'][key])
                label_dict[key].append(label_)

            # Load partial dataset for debug
            if self.debug and len(path_list) == 100: break
        
        labels = np.vstack(list(label_dict.values())).astype(np.int32).T
        return path_list, labels


    # Support function for imbalance training
    def get_imb(self):
        self.img_num_per_cls = self.get_cls_num(dataset_name='clothing1mpp', y_train=self.targets,
                                imb_factor=self.imb_factor)
        selected_idx = self.get_cls_idx(num_per_cls=self.img_num_per_cls, y_train=self.targets,
                             is_upsampling=self.is_upsampling)
        
        random.shuffle(selected_idx)
        # select from list 
        self.data = [self.data[idx] for idx in selected_idx]
        self.targets = self.targets[selected_idx]

    # Support function for imbalance training
    def get_cls_num(self, dataset_name, y_train, imb_factor=None):
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
        cls_num_dict ={'cifar10': 10, 'cifar100': 100,'clothing1mpp': 12}
        cls_num = cls_num_dict[dataset_name]
        img_max = len(list(np.where(np.array(y_train) == 0)[0]))
        if imb_factor is None:
            return [img_max] * cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            idx_this_class = list(np.where(np.array(y_train) == cls_idx)[0])
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(min(int(num), len(idx_this_class)))
        return img_num_per_cls
    
    # Support function for imbalance training
    def get_cls_idx(self, num_per_cls, y_train, is_upsampling=False):
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

if __name__ == '__main__':
    root = '/media/otter/cloth1m/version1'

    path_dict ={
        'root': '/media/otter/cloth1m/version1', # Root dir of the dataset that includes folders of images: Jacket, Shirt, etc.
        'full_label_json_path': '/media/otter/cloth1m/version1/labels_changeidx.json', # Json file that includes all the labels
        'split_path':'/media/otter/cloth1m/version1/labels/' # Dir includes to test_ids.pt, train_ids.pt, val_ids.pt
    }
    label_types = ['Class','Color','Material','Pattern']

    # Example usage
    batch_size = 64
    image_size = 256
    debug = False
    num_workers = 16

    train_set = Clothing1mPP(path_dict, image_size,debug=debug,split='train',label_types=label_types)
    pre_load = train_set.pre_load
    val_set = Clothing1mPP(path_dict, image_size,debug=debug,split='val',label_types=label_types,pre_load=pre_load)
    test_set = Clothing1mPP(path_dict, image_size,debug=debug,split='test',label_types=label_types,pre_load=pre_load)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    for i, (images, labels) in enumerate(train_loader):
        print("Train  batch: ", i, "Total Batch Count: ", len(train_loader))
        print(images.shape)
        print(labels.shape)
        # break

    for i, (images, labels) in enumerate(val_loader):
        print("Val  batch: ", i, "Total Batch Count: ", len(val_loader))
        print(images.shape)
        print(labels.shape)
        # break
    
    for i, (images, labels) in enumerate(test_loader):
        print("Test  batch: ", i, "Total Batch Count: ", len(test_loader))
        print(images.shape)
        print(labels.shape)
        # break
