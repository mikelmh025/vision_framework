import torch
import torchvision
import torchvision.transforms as transforms

def test_control(test_package, loop_type):
    if loop_type == 'default':
        return test_default(test_package)
    else:
        # We are using the same test loop
        return test_default(test_package)
   

def test_default(test_package):
    testloader, model, criterion, device, config \
        = test_package['testloader'], test_package['model'], test_package['criterion'], test_package['device'], test_package['config']
    
    correct, total = 0.0, 0.0
    for i, data in enumerate(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        # loss += criterion(outputs, labels).item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if config['test']['debug'] and i >= 5: break

    acc = correct / total * 100
    # loss = loss / len(testloader)
    return acc