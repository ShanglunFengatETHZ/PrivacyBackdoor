import torch
import torch.nn as nn
from opacus.utils.batch_memory_manager import BatchMemoryManager
from edit_vit import TransformerWrapper


# TODO: use the following codes to control the random values
# TODO: random.seed(seed_val)
# TODO: np.random.seed(seed_val)
# TODO: torch.manual_seed(seed_val)
# TODO: torch.cuda.manual_seed_all(seed_val)


def train_model(model, dataloaders, optimizer, num_epochs, device='cpu', verbose=False, logger=None, is_debug=False):
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

                if isinstance(model, TransformerWrapper) and verbose and phase == 'train':
                    r_delta_wt_conv, r_delta_bs_conv = model.show_perturbation()
                    wt_change, bs_change = model.show_weight_bias_change()
                    if logger is None:
                        print(f'conv delta weight:{r_delta_wt_conv}, conv delta bias:{r_delta_bs_conv}')
                        print(f'bkd delta weight:{wt_change}')
                        print(f'bkd delta bias:{bs_change}')
                        print(f'number of outliers: {len(model.backdoor_activation_history)}')
                    else:
                        logger.info(f'conv delta weight:{r_delta_wt_conv}, conv delta bias:{r_delta_bs_conv}')
                        logger.info(f'bkd delta weight:{wt_change}')
                        logger.info(f'bkd delta bias:{bs_change}')
                        logger.info(f'number of outliers: {len(model.backdoor_activation_history)}')

                if verbose and is_debug:
                    print(f'phase:{phase}, max logits:{outputs.max()}, min logits:{outputs.min()}, '
                          f'variance:{outputs.var(dim=0)}')

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.eq(labels.data))

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
    return acc, epsilon, delta


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


def text_debug(monitor, debug_dict=None, logger=None, step=None):
    print_period = debug_dict.get('print_period', 1)
    delta_bkd_bias_printable, delta_bkd_bias, delta_bkd_estimate_printable, delta_bkd_estimate = monitor.get_backdoor_change()
    if step % print_period == 0:
        if logger is None:
            print(f'*** STEP {step}: delta bias ***')
        else:
            logger.info(f'*** STEP {step}: delta bias ***')
        for j in range(len(delta_bkd_bias_printable)):
            if logger is None:
                print(f'sequence{j}:{delta_bkd_bias_printable[j]}')
            else:
                logger.info(f'sequence{j}:{delta_bkd_bias_printable[j]}')

        if logger is None:
            print(f'*** STEP {step}: delta estimate ***')
        else:
            logger.info(f'*** STEP {step}: delta estimate ***')
        for j in range(len(delta_bkd_estimate_printable)):
            if logger is None:
                print(f'sequence{j}:{delta_bkd_estimate_printable[j]}')
            else:
                logger.info(f'sequence{j}:{delta_bkd_estimate_printable[j]}')

    neg_grad_flow_strategy = debug_dict.get('negative_gradient_flow_strategy', 'reject')

    delta_bkd_bias = monitor.d1tod2(delta_bkd_bias)
    exist_negative_grad_flow = torch.logical_not(torch.all(delta_bkd_bias <= 0.0))
    match neg_grad_flow_strategy:
        case 'reject':
            assert not exist_negative_grad_flow, 'there are negative gradient flow'
        case 'report':
            neg_grad_flow_this_sequence = (torch.sum(delta_bkd_bias > 0.0, dim=-1) > 0)
            for j in range(len(delta_bkd_bias)):
                if neg_grad_flow_this_sequence[j]:
                    if logger is None:
                        print(f'step:{step}, sequence{j}: {delta_bkd_bias[j].tolist()}')
                    else:
                        logger.info(f'step:{step}, sequence{j}:{delta_bkd_bias[j].tolist()}')
        case _:
            pass


def text_train(model, train_dataloader, optimizer, device='cpu',
               logger=None, is_debug=False, debug_dict=None, monitor=None):
    total_train_loss = 0
    model.train()
    is_correct_lst = []

    for step, batch in enumerate(train_dataloader):
        if step % 20 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}'.format(step, len(train_dataloader)))
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)

        model.zero_grad()
        outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        loss, logits, hidden_states = outputs['loss'], outputs['logits'], outputs['hidden_states']
        if is_debug and monitor is not None:
            text_debug(monitor, debug_dict=debug_dict, logger=logger, step=step)

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
    return acc, avg_train_loss


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



