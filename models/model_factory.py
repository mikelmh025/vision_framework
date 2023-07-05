from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg11, vgg13, vgg16, vgg19
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

import torch
from torch import nn

import importlib

def get_model(name, full_package=None, pretrained=True, num_classes=1000):
    print('==> Building model...')
    assert full_package is not None, 'full_package must be provided, else something wrong with the preivous code.'

    if 'custom' not in name.lower():
        model = get_model_backbone(name, pretrained=pretrained)
        # classification task
        model = modify_classifier(name, model, num_classes)
    else:
        try:
            # TODO Dynamically import the self-defined model
            model_name = name.split('_')[1]
            model_module = importlib.import_module(model_name.lower())
            model = model_module.Coconet(num_classes=num_classes)
        except (ModuleNotFoundError, AttributeError):
            print('Model not recognized.')
    
    full_package['model'] = model
    return full_package


def modify_classifier(name, model, num_classes):
    if 'resnet' in name.lower():
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vgg' in name.lower():
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif 'inception' in name.lower():
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vit' in name.lower():
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
        model.num_classes = num_classes
    else:
        print('We don\'t have this model yet.')
    
    return model


def get_model_backbone(name, pretrained=True):
    if 'resnet' in name.lower():
        if '18' in name.lower():
            model = resnet18(weights='IMAGENET1K_V1')
        elif '34' in name.lower():
            model = resnet34(weights='IMAGENET1K_V1')
        elif '50' in name.lower():
            model = resnet50(weights='IMAGENET1K_V1')
        elif '101' in name.lower():
            model = resnet101(weights='IMAGENET1K_V1')
        elif '152' in name.lower():
            model = resnet152(weights='IMAGENET1K_V1')


        
    elif 'vgg' in name.lower():
        if '11' in name.lower():
            model = vgg11(weights='IMAGENET1K_V1')
        elif '13' in name.lower():
            model = vgg13(weights='IMAGENET1K_V1')
        elif '16' in name.lower():
            model = vgg16(weights='IMAGENET1K_V1')
        elif '19' in name.lower():
            model = vgg19(weights='IMAGENET1K_V1')
        
    
    # input size: 224x224
    elif 'vit' in name.lower():
        if 'base' in name.lower():
            if '16' in name.lower():
                model = vit_b_16(weights='IMAGENET1K_V1')
            elif '32' in name.lower():
                model = vit_b_32(weights='IMAGENET1K_V1')
        elif 'large' in name.lower():
            if '16' in name.lower():
                model = vit_l_16(weights='IMAGENET1K_V1')
            elif '32' in name.lower():
                model = vit_l_32(weights='IMAGENET1K_V1')

    else:
        print('We don\'t have this model yet.')
        

    return model


if __name__ == "__main__":
    test_input = torch.randn(4, 3, 224, 224)
    name_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
    name_list += ['vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224', 'vit_large_patch32_224']
    name_list += ['custom_coconet']


    for name in name_list:
        model = get_model(name, full_package={}, pretrained=True, num_classes=10)['model']
        print('get model: ', name)
        out = model(test_input)#[0]
        print(out.shape)
        print('-------------------')
        if out.shape[1] != 10:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        