import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from tools import setdiff1d


def load_dataset(root, dataset):
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root, train=True, transform=transforms.ToTensor(), download=False)
        test_dataset = datasets.CIFAR10(root, train=False, transform=transforms.ToTensor(), download=False)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root, train=True, transform=transforms.ToTensor(), download=False)
        test_dataset = datasets.CIFAR100(root, train=False, transform=transforms.ToTensor(), download=False)
    elif dataset == 'imagenet':
        train_dataset = datasets.ImageNet(root, split='train', transform=transforms.ToTensor(), download=False)
        test_dataset = datasets.ImageNet(root, split='val', transform=transforms.ToTensor(), download=False)

    return train_dataset, test_dataset


def get_subdataset(ds, p=0.5, random_seed=12345678):
    assert 0.0 <= p <= 1.0, 'invalid proportion parameter'
    torch.manual_seed(random_seed)
    size = len(ds)

    subsize = int(size * p)
    idx = torch.multinomial(input=torch.ones(size), num_samples=subsize)  # input of multimomial is probability
    idx_complement = setdiff1d(size, idx)

    ds_sub0 = data.Subset(ds, idx)
    ds_sub1 = data.Subset(ds, idx_complement)
    return ds_sub0, ds_sub1


def get_dataloader(ds0, ds1, batch_size, num_workers):
    ds0_loader = data.DataLoader(dataset=ds0, batch_size=batch_size, shuffle=True, pin_memory=True,
                                   num_workers=num_workers)

    ds1_loader = data.DataLoader(dataset=ds1, batch_size=batch_size, shuffle=False, pin_memory=True,
                                  num_workers=num_workers)
    return ds0_loader, ds1_loader