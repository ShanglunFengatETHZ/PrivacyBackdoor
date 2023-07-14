import torch
import torch.nn as nn
from torch.optim import SGD
import copy


def get_optimizer(model, lr=0.1, heads_factor=None, linear_probe=False):
    if heads_factor is None:
        params = model.parameters()
        return SGD(params=params, lr=lr)
    else:
        params_encoder = [param for name, param in model.named_parameters() if name not in ['heads.weight', 'heads.bias']]
        params_fc = model.heads.parameters()
        if linear_probe:
            return SGD([{'params':params_fc, 'lr': lr * heads_factor}])
        return SGD([{'params': params_encoder, 'lr': lr}, {'params':params_fc, 'lr': lr * heads_factor}])


def train_model(model, dataloaders, optimizer, num_epochs, device, verbose=False,
                toy_model=True, direct_resize=None, logger=None):
    # only adjust device in this function
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if logger is not None:
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
                if direct_resize is not None:
                    big_inputs = torch.zeros(inputs.shape[0], inputs.shape[1], direct_resize, direct_resize)
                    big_inputs[:, :, 0:inputs.shape[2], 0:inputs.shape[3]] = inputs
                    inputs = big_inputs

                if logger is None:
                    print(f'batch {i}')
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward propagation
                    outputs = model(inputs)
                    model_this_step = copy.deepcopy(model)
                    loss = loss_func(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        # backward propagation
                        loss.backward()
                        optimizer.step()

                        if toy_model:
                            if verbose and logger is not None:
                                logger.info(model.backdoor.registrar.print_update_this_step())
                            model.backdoor.store_hooked_fish(inputs)

                r_delta_wt_conv, r_delta_bs_conv = model.show_perturbation()
                print(f'conv delta weight:{r_delta_wt_conv}, conv delta bias:{r_delta_bs_conv}')
                wt_change, bs_change = model.show_weight_bias_change()
                print(f'bkd max logits:{outputs.max()}')
                #print(f'bkd delta weight:{wt_change}')
                #print(f'bkd delta bias:{1000*bs_change}')

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.eq(labels.data))

                # debug

            # debug
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
            print(epoch_loss)
            print(epoch_acc)
            if logger is not None:
                logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if toy_model and logger is not None:
                if model.backdoor.registrar.is_log:
                    logger.info('activation times for each door'+str(model.backdoor.registrar.valid_activate_freq.tolist()))
                    logger.info('doors that been activated more than once at a time'+str(model.backdoor.registrar.is_mixture.tolist()))

    return model.to('cpu')
