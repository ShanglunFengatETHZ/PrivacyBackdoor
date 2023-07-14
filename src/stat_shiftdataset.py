from data import load_dataset, get_dataloader, get_subdataset
from tools import dl2tensor
from tools import find_different_classes
import torch
import matplotlib.pyplot as plt


def calculate_practical_similarity(tr_imgs, bait_imgs, noise_scaling=3.0, is_cos_similarity=False, is_centralize=True, rs=137):
    assert tr_imgs.shape[1:] == bait_imgs.shape[1:], f'training images:{tr_imgs.shape[1:].numel()}, testing images:{bait_imgs.shape[1:].numel()}'

    num_tr, num_bait = len(tr_imgs), len(bait_imgs)
    num_fts = tr_imgs.shape[1:].numel()

    tr_imgs_flatten = tr_imgs.reshape(num_tr, -1)
    bait_imgs_flatten = bait_imgs.reshape(num_bait, -1)

    torch.manual_seed(rs)
    noise = noise_scaling * torch.randn(num_fts)
    tr_inputs = tr_imgs_flatten + noise
    bait_inputs = bait_imgs_flatten + noise
    # bait_inputs = torch.randn_like(bait_imgs_flatten)

    if is_centralize:
        bait_inputs = bait_inputs - bait_inputs.mean(dim=1, keepdim=True)
        tr_inputs = tr_inputs - tr_inputs.mean(dim=1, keepdim=True)
    bait_inputs_normalized = bait_inputs / bait_inputs.norm(dim=1, keepdim=True)

    if is_cos_similarity:
        tr_inputs = tr_inputs / tr_inputs.norm(dim=1, keepdim=True)

    return bait_inputs_normalized @ tr_inputs.t()


def calculate_gaussian_similarity(tr_imgs, bait_imgs, noise_scaling, is_ortho=False):
    assert tr_imgs.shape[1:] == bait_imgs.shape[1:], f'training images:{tr_imgs.shape[1:].numel()}, testing images:{bait_imgs.shape[1:].numel()}'

    num_tr, num_bait = len(tr_imgs), len(bait_imgs)
    num_fts = tr_imgs.shape[1:].numel()

    tr_imgs_flatten = tr_imgs.reshape(num_tr, -1)
    bait_imgs_flatten = bait_imgs.reshape(num_bait, -1)

    noise = noise_scaling * torch.randn(num_fts)
    tr_inputs = tr_imgs_flatten + noise
    print(tr_inputs.norm(dim=1).max())
    print(tr_inputs.norm(dim=1).min())
    bait_inputs = torch.randn_like(bait_imgs_flatten)

    tr_inputs = tr_inputs - tr_inputs.mean(dim=1, keepdim=True)
    bait_inputs_normalized = bait_inputs / bait_inputs.norm(dim=1, keepdim=True)
    if is_ortho:
        Q, R = torch.linalg.qr(bait_inputs_normalized)
        bait_inputs_normalized = Q
    return bait_inputs_normalized @ tr_inputs.t(), bait_inputs_normalized




def find_replicate_samples(similarity, q_thres):
    count = torch.zeros(similarity.shape[1])

    q = torch.quantile(similarity, q=q_thres, dim=-1, keepdim=False)
    for j in range(len(similarity)):
        similarity_this_bait = (similarity[j] > q[j])
        count += similarity_this_bait
    print(torch.nonzero(count > 1.5))
    print(count[torch.nonzero(count > 1.5)])
    print(len(torch.nonzero(count > 1.5)))


def show_similarity_stats(similarity, mode, q=None):
    # adjust printing format, print advanced statistics
    if mode == 'mean':
        print(torch.mean(similarity, dim=1).tolist())
    elif mode == 'median':
        print(torch.median(similarity, dim=1).tolist())
    elif mode == 'min':
        print(torch.min(similarity, dim=1).values.tolist())
    elif mode == 'quantile':
        quantile_lst = torch.quantile(similarity, q=torch.tensor(q), dim=-1).permute(1, 0).tolist()
        for j in range(len(quantile_lst)):
            print(quantile_lst[j])
    return None


def simulate_training(similarity, baits, q_thres=0.999, batch_size=64, T=100):
    num_train_imgs = similarity.shape[1]
    similarity_baits = similarity[baits]
    similarity_quantile = torch.quantile(similarity_baits, q=q_thres, dim=-1, keepdim=True)

    for j in range(T):
        activate_this_backdoor = []
        idx_this_batch = torch.multinomial(input=1.0 * torch.ones(num_train_imgs), num_samples=batch_size)
        activate_this_batch = (similarity_baits[:, idx_this_batch] > similarity_quantile)
        for k in range(len(similarity_baits)):
            activate_this_backdoor.append(idx_this_batch[activate_this_batch[k]].tolist())
        print(activate_this_backdoor)


def gray_plot(imgs, h, w, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    avg_mean = mean.mean()
    avg_std = std.mean()

    plt.figure(dpi=150)
    for k in range(h * w):
        k = k + 1
        plt.subplot(h, w, k)
        if k >= len(imgs):
            img = torch.zeros_like(imgs[0])
        else:
            img = imgs[k]

        img_real = img * avg_std + avg_mean

        plt.imshow(img_real, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    ds_path = '../../cifar10'
    tr_ds, bait_ds, _, _ = load_dataset(ds_path, 'cifar10', is_normalize=True)
    tr_ds, _ = get_subdataset(tr_ds, p=1.0)
    bait_ds, _ = get_subdataset(bait_ds, p=0.2)

    tr_dl, bait_dl = get_dataloader(tr_ds, batch_size=64, num_workers=1, ds1=bait_ds)
    tr_imgs, tr_labels = dl2tensor(tr_dl)
    bait_imgs, bait_labels = dl2tensor(bait_dl)

    tr_imgs_cut = tr_imgs[:, :, 8:16, 8:24]
    bait_imgs_cut = bait_imgs[:, :, 8:24, 8:16]

    gray_weight = torch.tensor([0.30, 0.59, 0.11]).reshape(1, 3, 1, 1)

    tr_gray_imgs = torch.sum(gray_weight * tr_imgs_cut, dim=1)
    bait_gray_imgs = torch.sum(gray_weight * bait_imgs_cut, dim=1)

    noise_scaling = 3.0
    qthres = 0.999

    # similarity = calculate_practical_similarity(tr_imgs=tr_gray_imgs, bait_imgs=bait_gray_imgs)
    # similarity, bait_inputs_normalized = calculate_gaussian_similarity(tr_imgs=tr_gray_imgs, bait_imgs=bait_gray_imgs, noise_scaling=3.0)
    # show the inner product between training images for determining whether the result is safe
    cos_similarity = calculate_practical_similarity(tr_imgs=tr_gray_imgs, bait_imgs=tr_gray_imgs, noise_scaling=noise_scaling, is_cos_similarity=True)
    print(cos_similarity.min())
    # epsilon=3.0, min x^T x^\prime = 0.355
    # epsilon=2.0 min x^T x^\prime = -0.0215
    # print(torch.min(cos_similarity))

    # gap_classes_per_bait = find_different_classes(similarity, tr_labels=tr_labels, q=qthres, is_sort=False, is_print=True)
    # good_baits = [info_bait['idx_bait'] for info_bait in gap_classes_per_bait[:20]]
    # find_replicate_samples(similarity[torch.tensor(good_baits)], q_thres=0.9999)
    # print(1)


    # show how strong will the replicate effect be
    # simulate_training(similarity, baits=good_baits, q_thres=qthres, batch_size=64, T=100)

    # show the angle between bait and images
    # show_similarity_stats(cos_similarity[good_baits], mode='min')

    # show the range of inner product to know what scaling do we need for encoder
    # show_similarity_stats(similarity[good_baits], mode='quantile', q=[0.001, 0.5, 0.9988, 0.999, 0.9999])

    # show the grayscale of training images
    # gray_plot(tr_gray_imgs[0:64], h=8, w=8)

    # x = torch.arange(start=0.99, end=1.0, step=0.001)
    # y = x - ((1.0 - x) * 0.005)
    #print(torch.quantile(tr_gray_imgs.reshape(-1), q=x, dim=-1) - torch.quantile(tr_gray_imgs.reshape(-1), q=y, dim=-1))

    # gray_sort = tr_gray_imgs.reshape(-1).sort()
    # print(torch.sum(tr_gray_imgs.reshape(-1) < -2.3997) / len(tr_gray_imgs.reshape(-1)))
    # plt.hist(tr_gray_imgs.reshape(-1))
    # plt.show()

    #tr_imgs_flat = tr_imgs.reshape(len(tr_imgs), -1)
    #tr_imgs_prime = tr_imgs.reshape(len(tr_imgs), -1)
    #tr_imgs_prime = tr_imgs_prime / tr_imgs_prime.norm(dim=1, keepdim=True)
    #u, sigma, vt = torch.linalg.svd(tr_imgs_prime.t() @ tr_imgs_prime)
    #zz = tr_imgs_flat @ vt[:100, :].t()
    #gap_classes_per_bait = find_different_classes(zz.t(), tr_labels=tr_labels, q=qthres, is_sort=False, is_print=True)
    #good_baits = [info_bait['idx_bait'] for info_bait in gap_classes_per_bait[:32]]
    #find_replicate_samples(zz.t()[torch.tensor(good_baits)], q_thres=0.9999)



    ###
    # print(sigma[:64])

    """
        for j in range(64):
        plt.subplot(8, 8, j+1)
        plt.hist(zz[:, j])
        print(torch.quantile(zz[:, j].reshape(-1), q=x, dim=-1))
        print(torch.quantile(zz[:,j].reshape(-1), q=x, dim=-1) - torch.quantile(zz[:,j].reshape(-1), q=y, dim=-1))
        print(zz[:,j].max())
        print('*************')
        plt.axis('off')
    plt.show()
    """


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    main()

