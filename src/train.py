import torch
import torch.nn as nn
from torch.optim import SGD


def get_optimizer(model, lr=0.1):
    params = model.parameters()
    return SGD(params=params, lr=lr)


def train_model(model, dataloaders, optimizer, logger, num_epochs, device, verbose=False):
    # only adjust device in this function
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        logger.info('Epoch {}'.format(epoch))

        for phase in ['train', 'val']:  # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            dataloader = dataloaders[phase]

            # Iterate over data.
            for i, this_batch in enumerate(dataloader):
                inputs, labels = this_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward propagation
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        # backward propagation
                        loss.backward()
                        optimizer.step()
                        if verbose:
                            logger.info(';'.join([str(lst) for lst in model.backdoor._update_last_step]))
                        model.backdoor.store_hooked_fish(inputs)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.eq(labels.data))

                # debug


            # debug
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logger.info('activation times for each door'+str(model.backdoor._activate_frequency.tolist()))
            logger.info('doors that been activated more than once at a time'+str(model.backdoor._is_mixture.tolist()))
            logger.info('upper bound of replicate images'+str(model.backdoor._total_replica_within_same_batch))
    return model.to('cpu')
