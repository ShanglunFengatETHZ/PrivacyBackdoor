import torch
import torch.nn as nn
from opacus.utils.batch_memory_manager import BatchMemoryManager
from edit_vit import ViTWrapper
from model_mlp import NativeMLP


# TODO: use the following codes to control the random values
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)


def train_model(model, dataloaders, optimizer, num_epochs, device='cpu', logger=None,
                is_debug=False, debug_dict=None):

    model = model.to(device)  # only adjust device in this function
    loss_func = nn.CrossEntropyLoss()
    print_log = print if logger is None else logger.info
    train_acc_lst = []
    test_acc_lst = []
    if isinstance(model, ViTWrapper):
        model.activate_registrar()
    for epoch in range(num_epochs):
        print_log('Epoch {}'.format(epoch))

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

                if is_debug:
                    print_period = debug_dict.get('print_period', 1)
                if isinstance(model, ViTWrapper) and is_debug and phase == 'train' and i % print_period == 0:
                    print_log(f'Epoch:{epoch}, Step:{i}')
                    r_delta_wt_conv, delta_wt_conv, r_delta_bs_conv = model.show_conv_perturbation()

                    delta_estimate, delta_bias = model.show_backdoor_change(is_printable=True)

                    print_log(f'conv relative delta weight:{r_delta_wt_conv}, conv delta weight:{delta_wt_conv}, conv delta bias:{r_delta_bs_conv}')
                    print_group_lst = []
                    for j in range(len(delta_estimate)):
                        print_group_lst.append(f'({delta_estimate[j]},{delta_bias[j]})')

                        if (j+1) % 8 == 0:
                            print_group = ','.join(print_group_lst)
                            print_log(f'(delta estimate, delta bias),{print_group}')
                            print_group_lst = []

                    print_log(f'number of outliers: {len(model.activation_history)}')

                #    if model.is_splice:
                #        for group in model.indices_grp:
                #            delta_estimate, delta_bias = model.show_backdoor_change(is_printable=True, output_indices=group, debug=True)
                #            print_log(f'Step:{i}, Group:{(group[0], group[-1])}')
                #            print_log(f'Delta estimate:{delta_estimate}')
                #            print_log(f'Delta bias:{delta_bias}')

                if isinstance(model, NativeMLP) and is_debug and phase == 'train' and i % print_period == 0:
                    print_log(f'batch:{i}, delta bias:{model.show_backdoor_change()}')

                if is_debug and debug_dict.get('output_logit_stat', False):
                    std_lst = outputs.to('cpu').detach().std(dim=0).tolist()
                    std_lst = [round(std_bkd, 3) for std_bkd in std_lst]
                    print_log(f'step:{i}, phase:{phase}, max logits:{round(outputs.max().item(),3)}, min logits:{round(outputs.min().item(),3)}, std:{std_lst}')

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.eq(labels.data))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)

            print_log('Epoch:{} Phase:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_acc_lst.append(epoch_acc)
            if phase == 'val':
                test_acc_lst.append(epoch_acc)

        if isinstance(model, ViTWrapper):
            model.shutdown_registrar()
    logger.info(f'Training Accuracy:{train_acc_lst}')
    logger.info(f'Test Accuracy:{test_acc_lst}')
    return model.to('cpu')

"""
    if hasattr(model, 'backdoor') and phase == 'train' and i % print_period == 0:
        if verbose:
            logger.info(model.backdoor.registrar.print_update_this_step())
                model.backdoor.store_hooked_fish(inputs)
     if hasattr(model, 'backdoor') and logger is not None:
        if model.backdoor.registrar.is_log:
            logger.info('activation times for each door'+str(model.backdoor.registrar.valid_activate_freq.tolist()))
            logger.info('doors that been activated more than once at a time'+str(model.backdoor.registrar.is_mixture.tolist()))
"""


def dp_train_by_epoch(model, train_loader, optimizer, privacy_engine, backdoor_registrar, epoch, delta=1e-5,
                      device='cpu', max_physical_batch_size=128, logger=None, use_inner_output=True):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    is_correct_lst = []
    # is_update_state = False

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
                # backdoor_registrar.collect_inner_state(inner_output)
            else:
                output = model(images)

            loss = criterion(output, target)

            _, preds = torch.max(output, 1)
            # measure accuracy and record loss
            is_correct = torch.eq(preds, target)

            losses.append(loss.item())
            is_correct_lst.append(is_correct)

            loss.backward()  # use _check_skip_next_step to know updates

            """
            if not optimizer._check_skip_next_step(pop_next=False):
                is_update_state = True
            """

            optimizer.step()

            """
            if is_update_state:
                model.update_state()
                model.backdoor_registrar.update_log_logical()
                is_update_state = False
            """
            if not optimizer._is_last_step_skipped:
                # p.summed_grad = None
                backdoor_registrar.update_grad_log(model)
                backdoor_registrar.update_v2class_log(model)

        epsilon = privacy_engine.get_epsilon(delta)

        is_correct_all = torch.cat(is_correct_lst)
        acc = torch.mean(is_correct_all.float()).item()
        losses = torch.tensor(losses)
        avg_loss = losses.mean().item()
    if logger is not None:
        logger.info(f"Epoch {epoch} Loss: {avg_loss:.4f} Acc@1: {acc:.4f} (ε = {epsilon:.3f}, δ = {delta}, noise_multiplier = {optimizer.noise_multiplier:.3f})")
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

            is_correct = torch.eq(preds, labels)
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
        monitor.extract_real_sequences(input_ids, hidden_states, logits, step)
        if is_debug and monitor is not None:
            text_debug(monitor, debug_dict=debug_dict, logger=logger, step=step)

        total_train_loss += loss.item()
        _, preds = torch.max(logits, 1)
        is_correct = torch.eq(preds, labels)
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
        is_correct = torch.eq(preds, labels)
        is_correct_lst.append(is_correct)

    avg_val_loss = total_eval_loss / len(evaluation_dataloader)
    is_correct_all = torch.cat(is_correct_lst)
    acc = torch.mean(is_correct_all.float()).item()
    return acc, avg_val_loss



