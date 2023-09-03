import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from tools import setdiff1d
from torch.utils.data import TensorDataset, random_split
import datasets as hgf


def load_dataset(root, dataset, is_normalize=False, resize=None, is_augment=False):
    # TODO: systemize this function
    if dataset == 'cifar100':
        transform_lst = []
        if is_augment:
            transform_lst.append(transforms.RandomHorizontalFlip())
            transform_lst.append(transforms.RandomRotation(20))
            transform_lst.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

        transform_lst.append(transforms.ToTensor())
        if is_normalize:
            transform_lst.append(transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)))
        train_dataset = datasets.CIFAR100(root, train=True, transform=transforms.Compose(transform_lst), download=False)
        test_dataset = datasets.CIFAR100(root, train=False, transform=transforms.Compose(transform_lst[-2:]), download=False)
        resolution = 32
        classes = 100
    elif dataset == 'imagenet':
        train_dataset = datasets.ImageNet(root, split='train', transform=transforms.ToTensor(), download=False)
        test_dataset = datasets.ImageNet(root, split='val', transform=transforms.ToTensor(), download=False)
        resolution = 224
        classes = 1000
    else:
        transform_lst = []
        if resize is not None:
            transform_lst.append(transforms.Resize(resize))
        transform_lst.append(transforms.ToTensor())
        if is_normalize:
            transform_lst.append(transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)))

        train_dataset = datasets.CIFAR10(root, train=True, transform=transforms.Compose(transform_lst), download=False)
        test_dataset = datasets.CIFAR10(root, train=False, transform=transforms.Compose(transform_lst), download=False)
        resolution = 32
        classes = 10

    return train_dataset, test_dataset, resolution, classes


def get_sentences_labels_from_dicts(dataset, text_key, label_key):
    m = len(dataset)
    sentences_lst = []
    labels_lst = []
    for j in range(m):
        info = dataset[j]
        text, label = info[text_key], info[label_key]
        sentences_lst.append(text)
        labels_lst.append(label)
    return sentences_lst, labels_lst


def cope_with_sentences(sentences, tokenizer=None, max_len=64, pad_to_max_length=True):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_len,
                                             pad_to_max_length=pad_to_max_length, return_attention_mask=True,
                                             return_tensors='pt')

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def load_text_dataset(root=None, dataset=None, tokenizer=None, max_len=64):
    if dataset == 'trec':
        train_dicts, test_dicts = hgf.load_dataset('trec', split='train'), hgf.load_dataset('trec', split='test')
        train_sentences, train_labels = get_sentences_labels_from_dicts(train_dicts, text_key='text', label_key='coarse_label')
        test_sentences, test_labels = get_sentences_labels_from_dicts(test_dicts, text_key='text', label_key='coarse_label')
        classes = 6
    else:
        train_sentences, train_labels = None, None
        test_sentences, test_labels = None, None
        classes = 2
        pass
    train_input_ids, train_attention_masks = cope_with_sentences(train_sentences, tokenizer=tokenizer, max_len=max_len)
    train_labels = torch.tensor(train_labels)
    test_input_ids, test_attention_masks = cope_with_sentences(test_sentences, tokenizer=tokenizer, max_len=max_len)
    test_labels = torch.tensor(test_labels)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    return train_dataset, test_dataset, classes


def get_subdataset(ds, p=0.5, random_seed=12345678):
    # TODO: use torch.utils.data.random_split
    if p is None:
        return ds, None
    else:
        assert 0.0 <= p <= 1.0, 'invalid proportion parameter'
        torch.manual_seed(random_seed)
        size = len(ds)

        subsize = int(size * p)
        idxs = torch.multinomial(input=torch.ones(size), num_samples=subsize)  # input of multi-nomial is probability
        idxs_complement = setdiff1d(size, idxs)

        ds_sub0 = data.Subset(ds, idxs)
        ds_sub1 = data.Subset(ds, idxs_complement)
        return ds_sub0, ds_sub1


def get_dataloader(ds0, batch_size, num_workers, ds1=None):
    # return one or two dataloaders
    ds0_loader = data.DataLoader(dataset=ds0, batch_size=batch_size, shuffle=True,
                                 pin_memory=True, num_workers=num_workers)

    if ds1 is not None:
        ds1_loader = data.DataLoader(dataset=ds1, batch_size=batch_size, shuffle=False,
                                     pin_memory=True, num_workers=num_workers) # shuffle v.s. sampler https://discuss.pytorch.org/t/samplers-vs-shuffling/73740
        return ds0_loader, ds1_loader
    else:
        return ds0_loader


def get_direct_resize_dataset(dataset, start_from=(0, 0), target_size=(224, 224), default_values=0.0):
    image_lst, label_lst = [], []
    for j in range(len(dataset)):
        image, label = dataset[j]
        canvas = default_values * torch.ones(3, target_size[0], target_size[1])
        large_image = canvas[:, start_from[0]: (start_from[0] + target_size[0]),
                      start_from[1]: (start_from[1] + target_size[1])]
        image_lst.append(large_image)
        label_lst.append(label)
    large_images = torch.stack(image_lst)
    labels = torch.tensor(label_lst)
    return TensorDataset(large_images, labels)
