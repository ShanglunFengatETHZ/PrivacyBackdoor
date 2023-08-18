import torch
import torch.nn as nn
from torch.optim import SGD
from opacus.utils.batch_memory_manager import BatchMemoryManager


def get_optimizer(model, lr=0.1, heads_factor=None, only_linear_probe=False):
    # TODO: abandon this function, to work in transformer, output parameters in different part for wrapper.
    if heads_factor is None:
        params = model.parameters()
        return SGD(params=params, lr=lr)
    else:
        params_encoder = [param for name, param in model.named_parameters() if name not in ['heads.weight', 'heads.bias']]
        params_fc = model.heads.parameters()
        if only_linear_probe:
            return SGD([{'params': params_fc, 'lr': lr}])
        return SGD([{'params': params_encoder, 'lr': lr}, {'params': params_fc, 'lr': lr * heads_factor}])

# TODO: use the following codes to control the random values
# TODO: random.seed(seed_val)
# TODO: np.random.seed(seed_val)
# TODO: torch.manual_seed(seed_val)
# TODO: torch.cuda.manual_seed_all(seed_val)


def train_model(model, dataloaders, optimizer, num_epochs, device='cpu', verbose=False, direct_resize=None, logger=None):
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
                if verbose:
                    print(f'batch {i}')
                inputs, labels = this_batch
                if direct_resize is not None:
                    big_inputs = torch.zeros(inputs.shape[0], inputs.shape[1], direct_resize, direct_resize)
                    big_inputs[:, :, 0:inputs.shape[2], 0:inputs.shape[3]] = inputs
                    inputs = big_inputs

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

                if hasattr(model, 'backdoor') and logger is not None and phase == 'train':
                    if verbose:
                        logger.info(model.backdoor.registrar.print_update_this_step())
                    model.backdoor.store_hooked_fish(inputs)

                if hasattr(model, 'show_perturbation') and verbose and phase == 'train':
                    r_delta_wt_conv, r_delta_bs_conv = model.show_perturbation()
                    print(f'conv delta weight:{r_delta_wt_conv}, conv delta bias:{r_delta_bs_conv}')
                if hasattr(model, 'show_weight_bias_change') and verbose and phase =='train':
                    wt_change, bs_change = model.show_weight_bias_change()
                    print(f'bkd delta weight:{wt_change}')
                    print(f'bkd delta bias:{bs_change}')

                if verbose:
                    print(f'phase:{phase}, max logits:{outputs.max()}, min logits:{outputs.min()}, variance:{outputs.var(dim=0)}')
                """
                with torch.no_grad():
                    if hasattr(model, 'model0'):
                        outputs_old = model.model0(inputs.double())
                        print(f'max old logits:{outputs_old.max()}, min old logits:{outputs_old.min()}, old variance:{outputs_old.var(dim=0)}')
                """

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.eq(labels.data))

            if hasattr(model, 'registrar') and hasattr(model.registrar, 'update'):
                model.registrar.update()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)
            if logger is not None:
                logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            else:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if hasattr(model, 'backdoor') and logger is not None:
                if model.backdoor.registrar.is_log:
                    logger.info('activation times for each door'+str(model.backdoor.registrar.valid_activate_freq.tolist()))
                    logger.info('doors that been activated more than once at a time'+str(model.backdoor.registrar.is_mixture.tolist()))

    return model.to('cpu')


def dp_train_by_epoch(model, train_loader, optimizer, privacy_engine, epoch, delta=1e-5,
                      device='cpu', max_physical_batch_size=128, logger=None, use_inner_output=True):
    # TODO: keep parameter epoch unchanged?
    # TODO: can this train function use for other DP-parameter combination
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    is_correct_lst = []
    is_update_state = False

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            if use_inner_output:
                output, inner_output = model(images)
                model.backdoor_registrar.collect_inner_state(inner_output)
            else:
                output = model(images)

            loss = criterion(output, target)

            _, preds = torch.max(output, 1)
            # measure accuracy and record loss
            is_correct = (preds == target)

            losses.append(loss.item())
            is_correct_lst.append(is_correct)

            loss.backward()  # use _check_skip_next_step to know updates

            if not optimizer._check_skip_next_step(pop_next=False):
                is_update_state = True
            optimizer.step()

            if is_update_state:
                model.update_state()
                model.backdoor_registrar.update_log_logical()
                is_update_state = False

        epsilon = privacy_engine.get_epsilon(delta)

        is_correct_all = torch.cat(is_correct_lst)
        acc = torch.mean(is_correct_all.float()).item()
        losses = torch.tensor(losses)
        avg_loss = losses.mean().item()
    if logger is not None:
        logger.info(f"Epoch {epoch} Loss: {avg_loss:.4f} Acc@1: {acc:.4f} (ε = {epsilon:.3f}, δ = {delta})")


def evaluation(model, test_loader, device, use_inner_output=True):
    model.eval()
    is_correct_lst = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_inner_output:
                outputs, _ = model(images)
            else:
                outputs = model(images)

            _, preds = torch.max(outputs, 1)

            is_correct = (preds == labels)
            is_correct_lst.append(is_correct)

    is_correct_all = torch.cat(is_correct_lst)
    acc = is_correct_all.float().mean().item()
    return acc


def text_train(model, train_dataloader, optimizer, device='cpu', logger=None, is_debug=False):
    total_train_loss = 0
    model.train()
    is_correct_lst = []

    for step, batch in enumerate(train_dataloader):
        if step % 20 == 0 and not step == 0:
            if logger is None:
                print('  Batch {:>5,}  of  {:>5,}'.format(step, len(train_dataloader)))
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)

        model.zero_grad()
        outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        if is_debug:
            hidden_states = outputs['hidden_states']
            #Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape

        total_train_loss += loss.item()
        _, preds = torch.max(logits, 1)
        is_correct = (preds == labels)
        is_correct_lst.append(is_correct)

        loss.backward()
        optimizer.step()
        # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    is_correct_all = torch.cat(is_correct_lst)
    acc = torch.mean(is_correct_all.float()).item()
    if logger is None:
        print(f'Accuracy:{acc}, Loss:{avg_train_loss}')


def text_evaluation(model, evaluation_dataloader, device='cpu'):
    model.eval()

    total_eval_loss = 0
    is_correct_lst = []

    for batch in evaluation_dataloader:
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
            loss, logits = outputs['loss'], outputs['logits']

        total_eval_loss += loss.item()
        _, preds = torch.max(logits, 1)
        is_correct = (preds == labels)
        is_correct_lst.append(is_correct)

    avg_val_loss = total_eval_loss / len(evaluation_dataloader)
    is_correct_all = torch.cat(is_correct_lst)
    acc = torch.mean(is_correct_all.float()).item()
    return acc, avg_val_loss



