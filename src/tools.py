# things like plot images and pint parameters.
import torch
import matplotlib.pyplot as plt
import math


def setdiff1d(n, idx):
    assert isinstance(n, int) and n > 0, 'invalid length'
    assert torch.sum(idx < n) == len(idx), 'Invalid indexes'

    all_idx = torch.arange(n, device=idx.device)
    combined = torch.cat((all_idx, idx))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]

    rearange = torch.randperm(len(difference)) # random permutation
    return difference[rearange]


def weights_generator(num_input, num_output, mode='uniform', is_normalize=True, images=None, c=1.0):
    if mode == 'gaussian':
        weights = torch.randn(num_input, num_output)
    elif mode == 'images':
        weights = torch.transpose(images)[:, num_output]
    elif mode == 'fixed_sparse':
        weights = torch.zeros(num_input, num_output)
        idx_nonzero = torch.randint(num_output, (num_input, ))
        weights[torch.arange(num_input), idx_nonzero] = c
    elif mode == 'classic':
        weights = torch.nn.init.xavier_normal_(torch.empty(num_input, num_output))
    else:
        weights = torch.rand(num_input, num_output)
    if is_normalize:
        return weights / weights.norm(dim=0, keepdim=True)


def predict(classifier, dataloader, device, topk=5):
    classifier = classifier.to(device)
    classifier.eval()

    correct = 0.0
    correct_top_k = 0.0

    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = classifier(X)

            y_hat = outputs.argmax(dim=1)
            _, y_hat_topk = torch.topk(outputs, topk)
            correct += torch.sum(y_hat.eq(y))
            correct_top_k += torch.eq(y_hat_topk, y.view(-1, 1)).sum()
    return size, correct, correct_top_k


def plot_recovery(images, bias=(0.0, 0.0, 0.0), scaling=(1.0, 1.0, 1.0)):
    # images is a list of tensors 3 * w * h
    num = len(images)
    h = math.ceil(math.sqrt(num) * 6 / 5)  # h > w
    w = math.ceil(num / h)
    fig, axs = plt.subplots(h, w)
    bias = torch.tensor(bias)
    scaling = torch.tensor(scaling)

    for j in range(num):
        image = images[j]
        image_revise = (image + bias) * scaling

        iw, ih = num // h, num % h
        axs[ih, iw].imshow(image_revise.permute(1, 2, 0))
