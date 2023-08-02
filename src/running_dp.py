import torch
import torch.nn as nn
import torch.utils.data as data
from tools import pass_forward, cal_set_difference_seq
from data import load_dataset, get_subdataset, get_dataloader
from adv_model import DiffPrvBackdoorRegistrar, DiffPrvBackdoorMLP, EncoderMLP
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
import torch.optim as optim
from train import dp_train_by_epoch, evaluation, train_model


def find_bait_by_metric(scores, num_target, bait_candiate=None, sill=0):
    values, indices = scores.topk(2, dim=0)  # values: num_trials * topk
    gap = values[:, 0] - values[:, 1]
    metric_sort = gap
    # metric_sort = gap / values[:,1]
    metric_sill = values[:, 1]
    _, indices_target = torch.sort(metric_sort, descending=True)
    indices_target = indices_target[metric_sill[indices_target] > sill]
    # TODO: better metric here, note that it may be possible that the inner proudct for a bait is all negative
    # indices_target = torch.tensor(random.sample(list(set(indices_target.tolist())), num_target))
    indices_target = indices_target[:num_target]

    return values[indices_target], indices[indices_target, 0], bait_candiate[indices_target]


def find_bait_by_match(scores, num_target, bait_candidate=None, target_img_indices=None):
    is_largest = scores[target_img_indices, torch.arange(len(target_img_indices))] >= scores.max(dim=0)[0]

    indices_candidate = torch.arange(len(target_img_indices))[is_largest]
    indices_candidate = indices_candidate[:num_target]
    values, _ = scores[:, indices_candidate].topk(2, dim=0)
    assert torch.norm(_ - target_img_indices[indices_candidate]) == 0, 'WRONG!'

    return values, target_img_indices[indices_candidate], bait_candidate[indices_candidate]


def target_sample_selector(model, dataset, num_target=1, approach='gaussian', num_trials=None, is_debug=False,
                           alpha=2.0, target_img_indices=None):
    #  return datatset \ {target}, target, weight, (lower bound, upper bound)
    dl = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    features, labels = pass_forward(model, dl, return_label=True)
    num_features = features.shape[1]
    num_trials = num_target if num_trials is None else num_trials

    if approach == 'gaussian':
        bait_candidate = torch.randn(num_trials, num_features)
        bait_candidate = bait_candidate / bait_candidate.norm(dim=1, keepdim=True)
        scores = features @ bait_candidate.t()  # num_samples * num_trials

        upperlowerbounds, target_img_indices, bait = find_bait_by_metric(scores, num_target=num_target, bait_candiate=bait_candidate)
    elif approach == 'self':
        bait_candidate = features[target_img_indices]
        bait_candidate = bait_candidate - alpha * bait_candidate.mean(dim=1, keepdim=True)
        bait_candidate = bait_candidate / bait_candidate.norm(dim=1, keepdim=True)
        scores = features @ bait_candidate.t()
        upperlowerbounds, target_img_indices, bait = find_bait_by_match(scores, num_target=num_target, bait_candidate=bait_candidate, target_img_indices=target_img_indices)
    else:
        pass
    complement_indices = cal_set_difference_seq(n=len(dataset), indices=target_img_indices)
    dataset_left = data.Subset(dataset, complement_indices)

    eps = 0.0
    targets_img = []
    for j in range(num_target):
        with torch.no_grad():
            img = dataset[target_img_indices[j]].unsqueeze()
            ft = model(img)
            similarity = ft @ bait.t()
            eps += (similarity - upperlowerbounds[j, 0])**2
        targets_img.append(img)
    eps = torch.sqrt(eps)
    if is_debug:
        print(f'eps:{eps}')
    assert eps < 1e-4, 'the images indices, upper lower bounds and bait does not match'
    return dataset_left, (targets_img, labels[target_img_indices]), bait, upperlowerbounds


def set_threshold(upperlowerbounds, center=0.6, safe_margin=0.4):
    upper_bound, lower_bound = upperlowerbounds[:, 0], upperlowerbounds[:, 1]
    threshold = center * upper_bound + (1 - center) * lower_bound
    passing_threshold = (center - safe_margin) * upper_bound + (1 - center + safe_margin) * lower_bound
    return threshold, passing_threshold


def build_public_model(info_dataset, info_model, info_train, logger, save_path=None):
    ds_name, ds_path = info_dataset['NAME'], info_dataset['ROOT']
    is_normalize_ds = info_dataset.get('IS_NORMALIZE', True)
    train_dataset, test_dataset, resolution, num_classes = load_dataset(ds_path, ds_name, is_normalize=is_normalize_ds)

    # dp-sgd training - related hyper-parameters
    batch_size, learning_rate, num_epochs = info_train.get('BATCH_SIZE', 1024), info_train.get('LR', 0.1), info_train.get('EPOCHS', 10)
    device, num_workers, verbose = info_train.get('DEVICE', 'cpu'), info_train.get('NUM_WORKERS', 2), info_train.get('VERBOSE', False)

    # model architecture and initialization related hyper-parameters
    cnn_encoder_modules = info_model.get('CNN_ENCODER', None)
    cnn_encoder = nn.Sequential(*[eval('nn.' + cnn_module) for cnn_module in cnn_encoder_modules])
    mlp_sizes = info_model.get('MLP_SIZES', (256, 256))
    dropout = info_model.get('DROPOUT', None)
    classifier = EncoderMLP(cnn_encoder, mlp_sizes=mlp_sizes, input_size=(3, resolution, resolution),
                            num_classes=num_classes, dropout=dropout)
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    train_loader, test_loader = get_dataloader(ds0=train_dataset, ds1=test_dataset, batch_size=batch_size,
                                               num_workers=num_workers)
    dataloaders = {'train': train_loader, 'val': test_loader}
    train_model(classifier, dataloaders, optimizer, num_epochs=num_epochs, device=device, verbose=verbose, logger=logger)

    if save_path is not None:
        torch.save(classifier.state_dict(), save_path)


def build_dp_model(info_dataset, info_model, info_train, info_target, logger, save_path=None):
    # dataset-related hyper-parameters
    ds_name, ds_path = info_dataset['NAME'], info_dataset['ROOT']
    ds_train_subset, ds_estimate_subset = info_dataset.get('SUBSET', None), info_dataset.get('SUBSET_FOR_ESTIMATE', None)
    is_normalize_ds = info_dataset.get('IS_NORMALIZE', True)
    train_dataset, test_dataset, resolution, num_classes = load_dataset(ds_path, ds_name, is_normalize=is_normalize_ds)
    train_dataset, _ = get_subdataset(train_dataset, p=ds_train_subset, random_seed=12345678)

    # dp-sgd training - related hyper-parameters
    batch_size, max_physical_batch_size = info_train.get('BATCH_SIZE', 1024), info_train.get('MAX_PHYSICAL_BATCH_SIZE', 64)   # training
    is_with_epsilon = info_train.get('IS_WITH_EPSILON', False)
    max_grad_norm, epsilon, delta, noise_multiplier = info_train.get('MAX_GRAD_NORM', 1.0), info_train.get('EPSILON', 2.0), info_train.get('DELTA', 1e-5), info_train.get('NOISE_MULTIPLIER', 1.0)
    learning_rate, num_epochs,  = info_train.get('LR', 0.1), info_train.get('EPOCHS', 10)
    device, num_workers, verbose = info_train.get('DEVICE', 'cpu'), info_train.get('NUM_WORKERS', 2), info_train.get('VERBOSE', False)
    # train_random_seed = info_train.get('RANDOM_SEED', 12345678)

    # model architecture and initialization related hyper-parameters
    cnn_encoder_modules = info_model.get('CNN_ENCODER', None)
    cnn_encoder = nn.Sequential(*[eval('nn.' + cnn_module) for cnn_module in cnn_encoder_modules])
    mlp_sizes = info_model.get('MLP_SIZES', (256, 256))
    indices_bkd_u, indices_bkd_v = torch.tensor(info_model.get('INDICES_BKD_U', [-1])), torch.tensor(info_model.get('INDICES_BKD_V', [-1]))
    encoder_scaling_module_idx = info_model.get('ENCODER_SCALING_MODULE', -1)
    multipliers = info_model.get('MULTIPLIERS', {})

    has_membership = info_target.get('HAS_MEMBERSHIP', True)
    num_targets = info_target.get('NUM_TARGETS', 1)
    approach = info_target.get('APPROACH', 'gaussian')
    num_trials = info_target.get('NUM_TRIALS', 100)

    dataset_left, targets_img_label, bait, upperlowerbounds = target_sample_selector(cnn_encoder, dataset=train_dataset, num_target=num_targets, approach=approach, num_trials=num_trials,
                           is_debug=False, alpha=info_target.get('ALPHA', None), target_img_indices=info_target.get('TARGET_IMG', None))

    train_dataset = train_dataset if has_membership else dataset_left
    train_loader, test_loader = get_dataloader(ds0=train_dataset, ds1=test_dataset, batch_size=batch_size,
                                               num_workers=num_workers)
    threshold, passing_threshold = set_threshold(upperlowerbounds, center=0.6, safe_margin=0.4)

    backdoor_registrar = DiffPrvBackdoorRegistrar(indices_bkd_u=indices_bkd_u, indices_bkd_v=indices_bkd_v,
                                                  m_u = mlp_sizes[0], m_v=mlp_sizes[1], targets=targets_img_label[0],
                                                  labels=targets_img_label[1])
    classifier = DiffPrvBackdoorMLP(encoder=cnn_encoder, mlp_sizes=mlp_sizes, input_size=(3, resolution, resolution),
                                    num_classes=num_classes, backdoor_registrar=backdoor_registrar)
    classifier.vanilla_initialize(encoder_scaling_module_idx=encoder_scaling_module_idx, weights=bait, thresholds=threshold,
                                  passing_threshold=passing_threshold, factors=multipliers)

    errors = ModuleValidator.validate(classifier, strict=False)
    logger.info(errors)
    optimizer = optim.SGD(classifier.mlp_parameters(), lr=learning_rate)
    privacy_engine = PrivacyEngine()
    if is_with_epsilon:
        classifier, optimizer, train_loader = privacy_engine.make_private_with_epsilon(module=classifier, optimizer=optimizer,
            data_loader=train_loader, epochs=num_epochs, target_epsilon=epsilon, target_delta=delta, max_grad_norm=max_grad_norm)
    else:
        classifier, optimizer, train_loader = privacy_engine.make_private(module=classifier, optimizer=optimizer, data_loader=train_loader,
                                                            noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)
    logger.info(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

    for epoch in range(num_epochs):
        dp_train_by_epoch(classifier, train_loader, optimizer, privacy_engine, epoch=epoch, delta=delta, device=device,
                          max_physical_batch_size=max_physical_batch_size, logger=logger)

    classifier.update_state(epoch)
    test_acc = evaluation(classifier, test_loader, device=device)
    logger.info(f"\tTest set Acc: {test_acc:.4f}")

    if save_path is not None:
        if save_path[-4] == '.':
            assert save_path[-3:] == 'pth', 'WRONG saving suffix'
            prefix = save_path[:-4]
            weight_save_path = f'{prefix}_weight.pth'
            registrar_save_path = f'{prefix}_regis.pth'
        elif save_path[-3] == '.':
            assert save_path[-2:] == 'pt', 'WRONG saving suffix'
            prefix = save_path[:-3]
            weight_save_path = f'{prefix}_weight.pt'
            registrar_save_path = f'{prefix}_regis.pt'
        elif '.' not in save_path:
            weight_save_path = f'{save_path}_weight.pth'
            registrar_save_path = f'{save_path}_regis.pt'
        else:
            assert False, 'unparseable saving_path'

        torch.save(classifier.save_weight(), weight_save_path)
        torch.save(classifier.backdoor_registrar, registrar_save_path)


if __name__ == '__main__':
    ds_path = '../../cifar10'
    cnn_encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(32, 96, kernel_size=3, stride=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten()
    )
    tr_ds, test_ds, resolution, classes = load_dataset(ds_path, 'cifar10', is_normalize=True)
    tr_ds_left, target_img, bait, upperlowerbounds = target_sample_selector(cnn_encoder, dataset=tr_ds, num_target=1, approach='gaussian', num_trials=100, is_debug=True)

