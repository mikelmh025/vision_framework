import utils
import torch
import time

def train_epoch_default(train_package):
    # Things to save in train_package_last.pth
    keys_to_save = ['config','model','optimizer','epoch','running_vars','imbalance_info']
    
    train_package_keys = ['trainloader', 'valloader', 'model', 'criterion', 'optimizer', 'device', 'config','epoch']
    trainloader, valloader,model, criterion, optimizer, device, config, epoch = utils.parse_keys(train_package_keys, train_package)
    iterations_per_epoch = len(trainloader)

    save_model_dir = config['general']['save_model_dir']
    train_summary_writer, eval_summary_writer, fp_log_res = \
        train_package['summary_writers']['train'], train_package['summary_writers']['eval'], train_package['summary_writers']['fp_log_res']

    init_running_vars(train_package)
    model.train()
    running_loss = 0.0
    for it, data in enumerate(trainloader):
        it_global = epoch * iterations_per_epoch + it

        inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = inputs.float(), labels.long()
        if len(labels.shape)>1:
            labels = torch.squeeze(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if config['train']['print_every'] > 0 and it % config['train']['print_every'] == config['train']['print_every'] - 1:
            train_summary_writer.add_scalar('loss/train', loss.item(), global_step=it_global)
            info_str = 'It: {}, loss: {:.5f}, time elapsed: {:.3f}'.format(
                it_global, running_loss / config["train"]["print_every"], time.time() - train_package['running_vars']['t0'])
            running_loss = 0.0
            print(info_str)
            fp_log_res.write(info_str + '\n')

        if it_global > 0 and it_global % config['train']['eval_freq'] == 0:
            # Save train_package
            utils.save_train_package(train_package,save_model_dir + '/train_package_last.pth',keys_to_save)
            eval_acc = eval(train_package)
            eval_summary_writer.add_scalar('acc/eval', eval_acc, global_step=it_global)
            model.train()
            info_str = 'It: {}, eval_acc: {:.5f}, time elapsed: {:.3f}'.format(
                it_global, eval_acc, time.time() - train_package['running_vars']['t0'])
        
            print(info_str)
            fp_log_res.write(info_str + '\n')

        if config['train']['debug'] and it >= 5: break 

    
    train_package['it_global'] += iterations_per_epoch
    train_package['epoch'] += 1

    return running_loss / len(trainloader)


def init_running_vars(train_package):
    if 'running_vars' not in train_package:
        train_package['running_vars'] = {}
    train_package['running_vars']['t0'] = time.time()
    

def eval(test_package):
    valloader, model, criterion, device, config \
        = test_package['valloader'], test_package['model'], test_package['criterion'], test_package['device'], test_package['config']
    
    model.eval()
    correct, total = 0.0, 0.0
    for i, data in enumerate(valloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = inputs.float(), labels.long()
        if len(labels.shape)>1:
            labels = torch.squeeze(labels)
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    return acc