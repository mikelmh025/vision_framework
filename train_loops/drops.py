import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import loss_factory
import cvxpy as cp
import time
import os



    
def train_epoch_drops(train_package):
    # Things to save in train_package_last.pth
    keys_to_save = ['config','model','optimizer','epoch','running_vars','drops_info','imbalance_info']
    
    train_package_keys = ['trainloader', 'valloader', 'model', 'criterion', 'optimizer', 'device', 'config','epoch']
    trainloader, valloader,model, criterion, optimizer, device, config, epoch = utils.parse_keys(train_package_keys, train_package)

    if config['train']['resume_train'] != False:
        # TODO: Train some initial model and load it here. Verify that it works
        file_name_ = config['general']['save_root'].\
            replace(config['general']['save_root'].split('/')[-1],config['train']['resume_train'])+\
            '/model/train_package_last.pth'
        utils.load_train_package(file_name_,keys_to_load=keys_to_save,train_package=train_package)
        

    # testloader = train_package['testloader']
    train_summary_writer, eval_summary_writer, fp_log_res = \
        train_package['summary_writers']['train'], train_package['summary_writers']['eval'], train_package['summary_writers']['fp_log_res']
    iterations_per_epoch = len(trainloader)

    config_drops = config['train']['drops']
    save_model_dir = config['general']['save_model_dir']

    

    init_running_vars_drops(train_package)
    
    model.train()
    running_loss = 0.0
    for it, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        it_global = epoch * iterations_per_epoch + it
        
        if it_global > 0 and it_global % config_drops['n_it_update'] == 0 and config['train']['loss_type'] == 'drops':
            # update g_y, lambd, and alpha_y w.r.t. val trainloader
            train_package = drops_updateg(train_package)
            # train_package = loss_factory.get_loss_function(config['train']['loss_type'],train_package) # Update loss function. But not really using it
            info_str = 'it_global: {}, g_y: {}, lambda: {:.4f}'.format(it_global, utils.tensor_to_string(train_package['drops_info']['g_y'].clone().detach().cpu()), \
                                                                train_package['drops_info']['lambd'])
            print(info_str)
            fp_log_res.write(info_str + '\n')

        if it_global % 100 == 0:
            train_summary_writer.add_scalar('loss/train', loss.item(), global_step=it_global)

            info_str = 'It: {}, loss: {:.5f}, time elapsed: {:.3f}'.format(
                it_global, loss.item(), time.time() - train_package['running_vars']['t0'])
            print(info_str)
            fp_log_res.write(info_str + '\n')

        
        if config_drops['dro_div'] == 'reverse-kl':
            eps_list = [i/2 for i in range(21)]
            eps_list[10] = eps_list[-1]
            eps_list[11] = config_drops['eps']
        elif config_drops['dro_div'] == 'kl':
            upp_v = np.log(config['data']['num_classes'])
            eps_list = [tmp_v * upp_v / 20 for tmp_v in range(21)]
            eps_list[10] = eps_list[-1]
            eps_list[11] = 1.0
        
        if it_global > 0 and it_global % config['train']['eval_freq'] == 0:
            # Save train_package
            utils.save_train_package(train_package,save_model_dir + '/train_package_last.pth',keys_to_save)
            # torch.save(train_package, save_model_dir + '/train_package_last.pth')
            
            loss_valid, acc_valid = drops_eval_metrics(train_package,eval_loader_type='valloader')
            loss_eval, acc_eval   = drops_eval_metrics(train_package,eval_loader_type='testloader')

            acc_valid_worst, _      = drops_eval_worst_metrics(train_package,eval_loader_type='valloader')
            acc_eval_worst, acc_all = drops_eval_worst_metrics(train_package,eval_loader_type='testloader')

            acc_eval_dro_list = [0] * len(train_package['running_vars']['best_acc_eval_at_valid_dro_list'])
            for kk in range(len(acc_eval_dro_list)):
                acc_eval_dro_list[kk] = drops_eval_dro_metrics(train_package,eps_list[kk],eval_loader_type='testloader')

                # acc_eval_dro_list[kk] = drops_eval_dro_metrics(eval_ds, eps_list[kk],
                #                                         g_y, samples_per_cls)
            acc_eval_dro_sel = drops_eval_dro_metrics(train_package,eps_list[11],eval_loader_type='testloader')
            acc_valid_dro_sel = drops_eval_dro_metrics(train_package,eps_list[11],eval_loader_type='valloader')
            
            # save model
            if acc_valid > train_package['running_vars']['best_acc_valid']:
                train_package['running_vars']['best_acc_valid'] = acc_valid
                train_package['running_vars']['best_acc_eval_at_valid'] = acc_eval
                torch.save(model.state_dict(), save_model_dir + '/mean_best.pth')
                _ = save_logits(train_package, file_name=save_model_dir + '/val_mean_best_', eval_loader_type='valloader')
                _ = save_logits(train_package, file_name=save_model_dir + '/test_mean_best_', eval_loader_type='testloader')

            if acc_valid_worst > train_package['running_vars']['best_acc_valid_worst']:
                train_package['running_vars']['best_acc_valid_worst'] = acc_valid_worst
                train_package['running_vars']['best_acc_eval_at_valid_worst'] = acc_eval_worst
                torch.save(model.state_dict(), save_model_dir + '/worst_best.pth')
                _ = save_logits(train_package, file_name=save_model_dir + '/val_worst_best_', eval_loader_type='valloader')
                _ = save_logits(train_package, file_name=save_model_dir + '/test_worst_best_', eval_loader_type='testloader')

            if acc_valid_dro_sel > train_package['running_vars']['best_acc_valid_dro_sel']:
                torch.save(model.state_dict(), save_model_dir + '/dro_best.pth')
                _ = save_logits(train_package, file_name=save_model_dir + '/val_dro_best_', eval_loader_type='valloader')
                _ = save_logits(train_package, file_name=save_model_dir + '/test_dro_best_', eval_loader_type='testloader')
                train_package['running_vars']['best_acc_valid_dro_sel'] = acc_valid_dro_sel
                for kk in range(len(train_package['running_vars']['best_acc_eval_at_valid_dro_list'])):
                    train_package['running_vars']['best_acc_eval_at_valid_dro_list'][kk] = acc_eval_dro_list[kk]
                train_package['running_vars']['best_acc_eval_at_valid_dro_sel'] = acc_eval_dro_sel

            
            eval_summary_writer.add_scalar('loss/valid', loss_valid, global_step=it_global)
            eval_summary_writer.add_scalar('loss/eval', loss_eval, global_step=it_global)
            eval_summary_writer.add_scalar('acc/valid', acc_valid, global_step=it_global)
            eval_summary_writer.add_scalar('acc/eval', acc_eval, global_step=it_global)
            eval_summary_writer.add_scalar('acc/best_eval_at_valid', train_package['running_vars']['best_acc_eval_at_valid'], global_step=it_global)
            eval_summary_writer.add_scalar('acc/valid_worst', acc_valid_worst, global_step=it_global)
            eval_summary_writer.add_scalar('acc/eval_worst', acc_eval_worst, global_step=it_global)
            eval_summary_writer.add_scalar('acc/best_eval_at_valid_worst', train_package['running_vars']['best_acc_eval_at_valid_worst'], global_step=it_global)

            for kk in range(len(train_package['running_vars']['best_acc_eval_at_valid_dro_list'])):
                eval_summary_writer.add_scalar(f'acc/best_eval_at_valid_dro_{kk}', train_package['running_vars']['best_acc_eval_at_valid_dro_list'][kk], global_step=it_global)

            eval_summary_writer.add_scalar('acc/best_eval_at_valid_dro_sel', train_package['running_vars']['best_acc_eval_at_valid_dro_sel'], global_step=it_global)
            info_str = (
                'It: {}, Valid loss: {:.3f}, Valid acc: {:.3f}, Valid W-acc: {:.3f}, '
                'Eval loss: {:.3f}, Eval acc: {:.3f}, '
                'Best Valid acc: {:.3f}, Best Valid acc_worst: {:.3f}, '
                'Best Eval acc: {:.3f}, Best Eval acc_worst: {:.3f},'
                ).format(
                    it_global, loss_valid, acc_valid, acc_valid_worst,
                    loss_eval, acc_eval, train_package['running_vars']['best_acc_valid'], train_package['running_vars']['best_acc_valid_worst'],
                    train_package['running_vars']['best_acc_eval_at_valid'], train_package['running_vars']['best_acc_eval_at_valid_worst'])
            for kk in range(len(train_package['running_vars']['best_acc_eval_at_valid_dro_list'])):
                info_str += 'Best Eval acc_dro_{:.3f}: {:.3f}'.format(
                    eps_list[kk], train_package['running_vars']['best_acc_eval_at_valid_dro_list'][kk])
            info_str += (
                'Best Eval acc_dro_sel: {:.3f}, '
                'time elapsed: {:.3f}, Acc list: {}').format(
                    train_package['running_vars']['best_acc_eval_at_valid_dro_sel'], time.time() - train_package['running_vars']['t0'], acc_all)
            print(info_str)
            # TODO log file
            fp_log_res.write(info_str + '\n')


        

        if config['train']['print_every'] > 0 and it_global % config['train']['print_every'] == config['train']['print_every'] - 1:
            print(f'[epoch {epoch + 1}, global iter {it_global + 1}] loss: {running_loss / config["train"]["print_every"]}')
            running_loss = 0.0

        if config['train']['debug'] and it >= 5: break 

    train_package['it_global'] += iterations_per_epoch
    train_package['epoch'] += 1

    return running_loss / len(trainloader)

def init_running_vars_drops(train_package):
    if 'running_vars' not in train_package: train_package['running_vars'] = {}

    zero_keys = ['best_acc_valid','best_acc_eval_at_valid','best_acc_valid_worst','best_acc_eval_at_valid_worst']
    zero_keys += ['best_acc_valid_dro_sel','best_acc_eval_at_valid_dro_sel']

    list_zero_keys = ['best_acc_eval_at_valid_dro_list']

    time_key = ['t0']

    for key in zero_keys:
        if key not in train_package: train_package['running_vars'][key] = 0
    
    for key in list_zero_keys:
        if key not in train_package: train_package['running_vars'][key] = [0] * 11 # Why 11? for eval. We want to show 11 different eval metrics
    
    for key in time_key:
        if key not in train_package: train_package['running_vars'][key] = time.time()
            
# @Jiaheng: Help me to tune this function
def drops_updateg(train_package):

    # Initialize Basic Variables
    train_package_keys = ['valloader', 'model', 'device', 'config']
    ds,model, device, config = utils.parse_keys(train_package_keys, train_package)

    samples_per_cls = train_package['imbalance_info']['samples_per_cls']
    
    # Initialize drops algorithm variables (dynamic variables)
    drops_info_keys = ['g_y', 'lambd', 'alpha_y', 'r_list']
    g_y, lambd, alpha_y, r_list = utils.parse_keys(drops_info_keys, train_package['drops_info'])
    
    g_y, alpha_y, r_list = g_y.to(device), alpha_y.to(device), r_list.to(device)

    # Initialize drops hyperparameters (static variables)
    config_drops = config['train']['drops']
    drops_config_keys = ['eps', 'eta_g', 'eta_lambda', 'weight_type','dro_div','g_type']
    delta,eta_g, eta_lambda,weight_type,dro_div, g_type = utils.parse_keys(drops_config_keys, config_drops)

    # Start updating
    loss_y_list = torch.tensor([0] * config['data']['num_classes']).float().to(device)
    num_y_list  = torch.tensor([0] * config['data']['num_classes']).to(device)

    model.eval()
    for it, data in enumerate(ds):
        images, y_true = data[0].to(device), data[1].to(device)
        y_pred = model(images)
        # Main steps for a mini-batch:
        #   Step 1: Prepare w_y (optional--for ce loss)
        #           For each class y in class list [1, ..., K],
        #           generate a sample weight w_y for the batch:
        #               [val_i for i in range(batch_size)]
        #               val_i =1 if the target for sample x_i is y;
       
        if weight_type == 'ce':
            # Create an identity matrix of size `num_classes` 
            eye_mat = np.eye(config['data']['num_classes'], dtype=int)
            # Map each value in `y_true` to a row in the identity matrix
            w_y_list = eye_mat[y_true.cpu()].T

        # Step 2: Update softmaxed of logits get model prediction (post-shift)
        #       softmax_y = softmax of logits
        #       model prediction is: argmax_y (alpha_y * softmax_y)
        y_pred_prob = F.softmax(y_pred, dim=-1)
        y_weighted_pred = y_pred_prob * alpha_y
        y_weighted_pred = y_weighted_pred / torch.sum(y_weighted_pred, dim=1, keepdim=True)
        arg_pred = torch.argmax(y_pred_prob * alpha_y, dim=1)

        # Step 3: Calculate L_y (per class loss), L is the 0-1 loss
        #         Get the 0-1 loss for the mini-batch with sample weight w_y;
        #                achieve with 0-1 loss:
        #                  get the index for class y: idx_y
        #                  L_y = 0-1_loss(y_pred[idx_y], y_true[idx_y])

        # TODO use tensor to do this for speed
        if weight_type == 'ce':
            tmp_loss = F.cross_entropy(y_weighted_pred, y_true,reduction='none')
            for i in range(config['data']['num_classes']):
                tmp = torch.sum(torch.tensor(w_y_list[i], dtype=torch.float32).to(device) * tmp_loss)
                num_y_list[i] += len(torch.where(y_true == i)[0])
                loss_y_list[i] += tmp.item()
        else:  # 0-1 loss for L
            acc_list = (arg_pred == y_true).int()
            for i in range(config['data']['num_classes']):
                idx = torch.where(y_true == i)[0]
                num_y_list[i] += len(idx)
                tmp_loss = 1 - torch.gather(acc_list, 0, idx.unsqueeze(1)).squeeze()
                tmp = tmp_loss.sum().float()
                loss_y_list[i] += tmp.item()


    # Continue with accumulated loss for the whole validation set
    loss_y_list = loss_y_list / torch.tensor(num_y_list, dtype=torch.float32)

    # Step 4: Get the Lagrangian constraint term
    # cons = lambda * (D(r, g) - delta)
    
    if dro_div == 'l2':
        div = torch.sum(torch.square(r_list - g_y))
    elif dro_div == 'l1':
        div = torch.sum(torch.abs(r_list - g_y))
    elif dro_div == 'reverse-kl':
        # D(g_y, r_list)
        tmp = torch.log((r_list + 1e-12) / (g_y + 1e-12))
        div = torch.sum(r_list * tmp)
    elif dro_div == 'kl':
        # D(g_y, r_list)
        tmp = torch.log((g_y + 1e-12) / (r_list + 1e-12))
        div = torch.sum(g_y * tmp)
    div = div.float()
    cons = lambd * (div - delta)

    # Step 5: Get the Lagrangian (with L_y list of size K)
    # Lagrangian = tf.reduce_sum(g_y_list * L_y_list) + cons
    lagrangian = (g_y * loss_y_list).sum().float()
    lagrangian -= cons


    # Step 6: EG update on g_y
    if g_type == 'eg':
        if dro_div == 'kl':
            part1 = eta_g * loss_y_list.float()
            log_r = torch.log(r_list + 1e-12).float()
            part2 = lambd * eta_g * log_r
            neu = part1 + part2
            neu += torch.log(g_y + 1e-12).float()
            den = lambd * eta_g + 1.
            g_y_updated = torch.exp(neu / den - 1.)
            g_y_updated /= torch.sum(g_y_updated)
        elif dro_div == 'reverse-kl':
            part1 = eta_g * lambd * torch.log(r_list + 1e-12).tolist() # Really to list?
            neu = g_y.float() + part1
            den = eta_g * loss_y_list.float()
            g_y_updated = neu / den
            g_y_updated /= torch.sum(g_y_updated)
    else:
        g_y_updated = r_list * torch.exp(loss_y_list / lambd)
        g_y_updated /= torch.sum(g_y_updated)
    # Step 7: EG update on lambda
    #         lambda <- lambda - eta_lambda * gradient(Lagrangian)_lambda
    lambd_updated = lambd + eta_lambda * cons
    #       Step 8: Update alpha_y
    #               For now: alpha_y = g_y/pi_y
    alpha_y_updated = g_y_updated / torch.tensor(samples_per_cls).float().to(device)
    alpha_y_updated *= torch.sum(torch.tensor(samples_per_cls).float())

    train_package['drops_info']['g_y'] = g_y_updated
    train_package['drops_info']['lambd'] = lambd_updated
    train_package['drops_info']['alpha_y'] = alpha_y_updated

    model.train()
    return train_package
    # return g_y_updated, lambd_updated, alpha_y_updated

def drops_eval_metrics(train_package, eval_loader_type='valloader'):
    # Initialize Basic Variables
    train_package_keys = [eval_loader_type, 'model', 'device', 'config','criterion']
    ds,model, device, config, criterion = utils.parse_keys(train_package_keys, train_package)

    samples_per_cls = train_package['imbalance_info']['samples_per_cls']

    # Initialize drops algorithm variables (dynamic variables)
    drops_info_keys = ['g_y', 'lambd', 'alpha_y', 'r_list']
    g_y, lambd, alpha_y, r_list = utils.parse_keys(drops_info_keys, train_package['drops_info'])

    
    # Initialize drops hyperparameters (static variables)
    config_drops = config['train']['drops']
    drops_config_keys = ['tau','eps']
    tau,eps = utils.parse_keys(drops_config_keys, config_drops)


    avg_loss = 0.
    num_samples = 0
    num_correct = 0
    model.eval()
    for it, data in enumerate(ds):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            logits = model(images)
        if config['train']['loss_type'] == 'drops':
            # a_y * exp(logit) is equivalent to log(a_y) + logit
            # final_alpha_y = [a / b for a, b in zip(g_y, samples_per_cls)]
            final_alpha_y = g_y / torch.tensor(samples_per_cls).float().to(device)
            final_alpha_y *= torch.sum(torch.tensor(samples_per_cls).float()).to(device)
            #  The sign is positive because class prior is in the denominator.
            logits = logits + tau * torch.log(final_alpha_y + 1e-12)

        if config['train']['loss_type'] in ['posthoc', 'posthoc_ce']:
            spc = torch.tensor(samples_per_cls, dtype=torch.float32).to(device)
            spc_norm = spc / torch.sum(spc)
            logits = logits - tau * torch.log(spc_norm + 1e-12).to(device)

        avg_loss += criterion(logits, labels).item() * len(labels)
        num_samples += len(labels)
        num_correct += torch.sum(torch.eq(torch.argmax(logits, dim=1, keepdim=False), labels)).item()

    avg_loss = avg_loss / len(ds)
    acc = num_correct / float(num_samples) * 100.
    return avg_loss, acc




    return

def drops_eval_worst_metrics(train_package, eval_loader_type='valloader'):
    # Initialize Basic Variables
    train_package_keys = [eval_loader_type, 'model', 'device', 'config','criterion']
    valid_ds,model, device, config, criterion = utils.parse_keys(train_package_keys, train_package)

    samples_per_cls = train_package['imbalance_info']['samples_per_cls']

    # Initialize drops algorithm variables (dynamic variables)
    drops_info_keys = ['g_y', 'lambd', 'alpha_y', 'r_list']
    g_y, lambd, alpha_y, r_list = utils.parse_keys(drops_info_keys, train_package['drops_info'])

    # Initialize drops hyperparameters (static variables)
    config_drops = config['train']['drops']
    drops_config_keys = ['tau','eps']
    tau,eps = utils.parse_keys(drops_config_keys, config_drops)

    num_samples = [0] * config['data']['num_classes']
    num_correct = [0] * config['data']['num_classes']

    model.eval()
    for it, data in enumerate(valid_ds):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            logits = model(images)
        if config['train']['loss_type'] == 'drops':
            # alpha_y * exp(logit) is equivalent to log(alpha_y) + logit            
            final_alpha_y = [a / b for a, b in zip(g_y, samples_per_cls)]
            final_alpha_y = torch.tensor(final_alpha_y, dtype=torch.float32)
            final_alpha_y *= torch.sum(torch.tensor(samples_per_cls, dtype=torch.float32))
            logits = logits + tau * torch.log(final_alpha_y + 1e-12).to(device)
        
        if config['train']['loss_type'] in ['posthoc', 'posthoc_ce']:
            spc = torch.tensor(samples_per_cls, dtype=torch.float32)
            spc_norm = spc / torch.sum(spc)
            logits = logits - tau * torch.log(spc_norm + 1e-12).to(device)

        acc_list = (torch.argmax(logits, dim=1) == labels).int()
        for cls_idx in range(config['data']['num_classes']):
            idx_this_class = torch.where(labels == cls_idx)[0]
            num_samples[cls_idx] += len(idx_this_class)
            acc_this_class = torch.gather(acc_list, 0, idx_this_class)
            num_correct[cls_idx] += torch.sum(acc_this_class).item()

    acc_per_cls = [a * 100. / b for a, b in zip(num_correct, num_samples)]
    return min(acc_per_cls), acc_per_cls

def drops_eval_dro_metrics(train_package,eps,eval_loader_type='valloader'):
    """Eval worst (with least num of train samples) group acc on test set."""
    # Initialize Basic Variables
    train_package_keys = [eval_loader_type, 'model', 'device', 'config','criterion']
    valid_ds,model, device, config, criterion = utils.parse_keys(train_package_keys, train_package)

    samples_per_cls = train_package['imbalance_info']['samples_per_cls']

    # Initialize drops algorithm variables (dynamic variables)
    drops_info_keys = ['g_y', 'lambd', 'alpha_y', 'r_list']
    g_y, lambd, alpha_y, r_list = utils.parse_keys(drops_info_keys, train_package['drops_info'])

    # Initialize drops hyperparameters (static variables)
    config_drops = config['train']['drops']
    drops_config_keys = ['tau','dro_div'] # Note that we are using a specific eps here 
    # TODO: Question: Update eps somehow?
    tau,dro_div = utils.parse_keys(drops_config_keys, config_drops)


    # get the value of base list
    base_list = [1 for _ in range(config['data']['num_classes'])]
    base_weight = torch.tensor(base_list, dtype=torch.float64)
    base_weight_norm = base_weight / torch.sum(base_weight)

    num_samples_cls = [0] * config['data']['num_classes']
    num_correct_cls = [0] * config['data']['num_classes']

    model.eval()
    for it, data in enumerate(valid_ds):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            logits = model(images)

        if config['train']['loss_type'] == 'drops':
            # alpha_y * exp(logit) is equivalent to log(alpha_y) + logit            
            final_alpha_y = [a / b for a, b in zip(g_y, samples_per_cls)]
            final_alpha_y = torch.tensor(final_alpha_y, dtype=torch.float32)
            final_alpha_y *= torch.sum(torch.tensor(samples_per_cls, dtype=torch.float32))
            logits = logits + tau * torch.log(final_alpha_y + 1e-12).to(device)

        if config['train']['loss_type'] in ['posthoc', 'posthoc_ce']:
            spc = torch.tensor(samples_per_cls, dtype=torch.float32).to(device)
            spc_norm = spc / torch.sum(spc).to(device)
            logits = logits - tau * torch.log(spc_norm + 1e-12).to(device)
        
        acc_list = (torch.argmax(logits, dim=1) == labels).int()
        
        for j in range(config['data']['num_classes']):
            idx = torch.where(labels == j)[0]
            num_samples_cls[j] += len(idx)
            acc_j = torch.gather(acc_list, 0, idx)
            num_correct_cls[j] += torch.sum(acc_j).item()
    acc_per_cls = [a / b for a, b in zip(num_correct_cls, num_samples_cls)]

    # get worst class weights from the optimization with divergence constrants

    base_weight_norm = base_weight_norm.detach().numpy()
    v = cp.Variable(config['data']['num_classes'])
    v.value = v.project(base_weight_norm)
    constraints = [v >= 0, cp.sum(v) == 1]
    if dro_div == 'l2':
        constraints.append(cp.sum(cp.square(v - base_weight_norm)) <= eps)
    elif dro_div == 'l1':
        constraints.append(cp.sum(cp.abs(v - base_weight_norm)) <= eps)
    elif dro_div == 'reverse-kl':
        # D(g, u)=sum_i u_i * [log(u_i) - log(g_i)],
        # g is the parameter v we aim to solve, u_i is the base_weight_norm.
        constraints.append(
            cp.sum(cp.kl_div(base_weight_norm, v)) <= eps)
    elif dro_div == 'kl':
        # D(g, u)=sum_i g_i * [log(g_i) - log(u_i)],
        # g is the parameter v we aim to solve, u_i is the base_weight_norm.
        constraints.append(
            cp.sum(cp.kl_div(v, base_weight_norm)) <= eps)
    # print(acc_per_cls)

    obj = cp.Minimize(cp.sum(cp.multiply(v, np.array(acc_per_cls))))
    prob = cp.Problem(obj, constraints)
    try:
        v.value = v.project(base_weight_norm)
        prob.solve(warm_start=True)
    except cp.error.SolverError:
        prob.solve(solver='SCS', warm_start=True)
    # print(v.value)
    # print(acc_per_cls)
    dro_acc = np.sum(np.multiply(v.value, np.array(acc_per_cls)))
    return dro_acc #.numpy()

def save_logits(train_package, file_name, eval_loader_type='valloader'):
    """save model prediction logits and labels."""

    train_package_keys = [eval_loader_type, 'model', 'device', 'config','criterion']
    ds,model, device, config, criterion = utils.parse_keys(train_package_keys, train_package)

    full_logits = []
    full_labels = []
    model.eval()
    for it, data in enumerate(ds):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            logits = model(images)
        full_logits.append(logits)
        full_labels.append(labels)

    full_logits = torch.cat(full_logits, dim=0)
    full_labels = torch.cat(full_labels, dim=0).to(torch.int64)

    # full_logits = torch.cat([torch.cat(logits, dim=0) for logits in full_logits], dim=0)
    # full_labels = torch.cat([torch.cat(labels, dim=0) for labels in full_labels], dim=0).to(torch.int64)

    new_name1 = file_name + 'logits.txt'
    if os.path.exists(new_name1):
        os.remove(new_name1)
    with open(new_name1, 'w') as logits_file:
        np.savetxt(logits_file, np.array(full_logits.cpu().numpy()))
    new_name2 = file_name + 'labels.txt'
    if os.path.exists(new_name2):
        os.remove(new_name2)
    with open(new_name2, 'w') as labels_file:
        np.savetxt(labels_file, np.array(full_labels.cpu().numpy()))
    return full_logits