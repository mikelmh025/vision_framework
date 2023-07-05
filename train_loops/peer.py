import utils

# TODO: implement peer training
def train_epoch_peer(train_package):
    train_package_keys = ['trainloader', 'valloader', 'model', 'criterion', 'optimizer', 'device', 'config','epoch']
    trainloader, valloader,model, criterion, optimizer, device, config, epoch = utils.parse_keys(train_package_keys, train_package)

    peerloader_x, peerloader_y = train_package['peerloader_x'], train_package['peerloader_y']
    peer_iter_x, peer_iter_y = iter(peerloader_x), iter(peerloader_y)

    iterations_per_epoch = len(trainloader)
    running_loss = 0.0
    for i, batch in enumerate(trainloader):

        peer_batch_x, peer_batch_y = next(peer_iter_x), next(peer_iter_y)
        peer_inputs_x, peer_labels_y = peer_batch_x[0].to(device), peer_batch_y[1].to(device)
        inputs, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        logits_peer = model(peer_inputs_x)

        loss = criterion(logits, logits_peer, labels, peer_labels_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if config['train']['print_every'] > 0 and i % config['train']['print_every'] == config['train']['print_every'] - 1:
            print(f'[epoch {epoch + 1}, iter {i + 1}] loss: {running_loss / config["train"]["print_every"]}')
            running_loss = 0.0

        if config['train']['debug'] and i >= 5: break 

    train_package['it_global'] += iterations_per_epoch
    train_package['epoch'] += 1

    return running_loss / len(trainloader)

