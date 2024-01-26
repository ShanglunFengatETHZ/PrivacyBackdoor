import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import datasets as hgf
import torchvision.datasets as datasets
from transformers import BertTokenizer


def load_dataset(root, dataset, is_normalize=False, resize=None, is_augment=False, inlaid=None):
    transform_lst_train, transform_lst_test = [], []
    if is_augment:
        transform_lst_train.append(transforms.RandomHorizontalFlip())
        transform_lst_train.append(transforms.RandomRotation(20))
        transform_lst_train.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    if resize is not None:
        transform_lst_train.append(transforms.Resize(size=(resize, resize)))
        transform_lst_test.append(transforms.Resize(size=(resize, resize)))

    transform_lst_train.append(transforms.ToTensor())
    transform_lst_test.append(transforms.ToTensor())
    inlaid_resolution = None
    if dataset == 'cifar100':
        if is_normalize:
            transform_lst_train.append(transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)))
            transform_lst_test.append(transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)))
        train_dataset = datasets.CIFAR100(root, train=True, transform=transforms.Compose(transform_lst_train), download=False)
        test_dataset = datasets.CIFAR100(root, train=False, transform=transforms.Compose(transform_lst_test), download=False)
        original_resolution = 32
        classes = 100
    elif dataset == 'imagenet':
        if is_normalize:
            transform_lst_train.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
            transform_lst_test.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        train_dataset = datasets.ImageNet(root, split='train', transform=transforms.Compose(transform_lst_train))
        test_dataset = datasets.ImageNet(root, split='val', transform=transforms.Compose(transform_lst_test))
        # train_dataset = datasets.ImageFolder(root=f'{root}/train', transform=transforms.Compose(transform_lst_train))
        # test_dataset = datasets.ImageFolder(root=f'{root}/val', transform=transforms.Compose(transform_lst_test))
        original_resolution = 224
        classes = 1000
    elif dataset == 'oxfordpet':
        if is_normalize:
            transform_lst_train.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
            transform_lst_test.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        train_dataset = datasets.OxfordIIITPet(root, split='trainval', target_types='category', transform=transforms.Compose(transform_lst_train))
        test_dataset = datasets.OxfordIIITPet(root, split='test', target_types='category', transform=transforms.Compose(transform_lst_test))
        original_resolution = None
        classes = 37
    elif dataset == 'caltech101':
        transform_lst_test.append(IfGray2Colorful())
        if is_normalize:
            transform_lst_test.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        dataset = datasets.Caltech101(root, target_type='category', transform=transforms.Compose(transform_lst_test))
        train_dataset, test_dataset = get_subdataset(dataset, p=0.666)
        original_resolution = None
        classes = 101
    else:
        if is_normalize:
            transform_lst_train.append(transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)))
            transform_lst_test.append(transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)))

        if inlaid is not None:
            transform_lst_train.append(DirectInlaid(**inlaid))
            transform_lst_test.append(DirectInlaid(**inlaid))
            inlaid_resolution = inlaid.get('target_size', (224, 224))[0]

        train_dataset = datasets.CIFAR10(root, train=True, transform=transforms.Compose(transform_lst_train), download=False)
        test_dataset = datasets.CIFAR10(root, train=False, transform=transforms.Compose(transform_lst_test), download=False)
        original_resolution = 32
        classes = 10

    if inlaid_resolution is not None:
        resolution = inlaid_resolution
    elif resize is not None:
        resolution = resize
    else:
        resolution = original_resolution

    return train_dataset, test_dataset, resolution, classes


class DirectInlaid(object):
    def __init__(self, start_from=(0, 0), target_size=(224, 224), default_values=0.0):
        self.start_from = start_from
        self.target_size = target_size
        self.default_values = default_values

    def __call__(self, img):
        large_image = self.default_values * torch.ones(3, self.target_size[0], self.target_size[1])
        large_image[:, self.start_from[0]: (self.start_from[0] + img.shape[1]), self.start_from[1]: (self.start_from[1] + img.shape[2])] = img
        return large_image


class IfGray2Colorful(object):
    def __init__(self):
        pass

    def __call__(self, img):
        num_channel = img.shape[0]
        if num_channel == 1:
            img = torch.cat([img, img, img], dim=0)
        return img



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
    elif dataset == 'trec50':
        train_dicts, test_dicts = hgf.load_dataset('trec', split='train'), hgf.load_dataset('trec', split='test')
        train_sentences, train_labels = get_sentences_labels_from_dicts(train_dicts, text_key='text', label_key='fine_label')
        test_sentences, test_labels = get_sentences_labels_from_dicts(test_dicts, text_key='text', label_key='fine_label')
        classes = 50
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
    if p is None:
        return ds, None
    else:
        assert 0.0 <= p <= 1.0, 'invalid proportion parameter'
        generator = torch.Generator().manual_seed(random_seed)
        ds_sub0, ds_sub1 = data.random_split(dataset=ds, lengths=[p, 1.0 - p], generator=generator)
        return ds_sub0, ds_sub1


def get_dataloader(ds0, batch_size, num_workers, ds1=None, shuffle=False, collate_fn=None):
    # return one or two dataloaders
    ds0_loader = data.DataLoader(dataset=ds0, batch_size=batch_size, shuffle=shuffle,
                                 pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

    if ds1 is not None:
        ds1_loader = data.DataLoader(dataset=ds1, batch_size=batch_size, shuffle=False,
                                     pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
        # shuffle v.s. sampler https://discuss.pytorch.org/t/samplers-vs-shuffling/73740
        return ds0_loader, ds1_loader
    else:
        return ds0_loader


if __name__ == '__main__':
    pass









