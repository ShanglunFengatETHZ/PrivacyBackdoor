import torch
import matplotlib.pyplot as plt
import argparse
import datetime
import math
from src.model import ToyConvEncoder
from src.data import load_dataset, get_subdataset, get_dataloader
from src.tools import pass_forward, moving_window_picker, plot_recovery


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the YOU want to test')

    parser.add_argument('--root', type=str)
    parser.add_argument('--dataset', default="cifar10", type=str)  # dataset
    parser.add_argument('--sample_subset', default=0.002, type=float)
    parser.add_argument('--bait_subset', default=0.0064, type=float)

    parser.add_argument('--out_resolution', default=4, type=int)

    parser.add_argument('--window_size', default=3, type=int)
    parser.add_argument('--padding', default=1, type=int)
    parser.add_argument('--stride', default=2, type=int)

    parser.add_argument('--rs', default=12345678)  # running
    parser.add_argument('--is_normalize', default=False, type=bool)

    return parser.parse_args()


def cal_self_similarity(images, is_normalize=False):
    # (num_sample, num_sub_images_height, num_sub_images_width, channels, height, width)
    # is normalize weights
    # TODO: should normalize?
    num_sample, num_sub_images_height, num_sub_images_width, channels, height, width = images.shape

    mats_similarity = []
    for idx_sample in range(num_sample):
        sub_images = images[idx_sample].reshape(num_sub_images_width * num_sub_images_height, channels * height * width)
        sub_images_as_inputs = sub_images

        if is_normalize:
            sub_images_as_weights = sub_images / sub_images.norm(dim=-1, keepdim=True)
        else:
            sub_images_as_weights = sub_images

        mat_similarity = sub_images_as_inputs @ sub_images_as_weights.t()
        mats_similarity.append(mat_similarity)
    return mats_similarity


def cal_diff_similarity(sub_images_sample, sub_images_bait, is_normalize=False):
    # num_bait, num_sample, num_sub_images_per_bait, num_sub_images_per_sample, similarity()
    num_sample, num_sub_images_height, num_sub_images_width, channels, height, width = sub_images_sample.shape
    num_bait = sub_images_bait.shape[0]
    sub_images_sample = sub_images_sample.reshape(num_sample, num_sub_images_height * num_sub_images_width, channels * height * width)
    sub_images_bait = sub_images_bait.reshape(num_bait, num_sub_images_height * num_sub_images_width, channels * height * width)

    if is_normalize:
        sub_images_bait = sub_images_bait / sub_images_bait.norm(dim=-1, p=2, keepdim=True)
        sub_images_sample = sub_images_sample / sub_images_sample.norm(dim=-1, p=2, keepdim=True)

    similarities = []
    for idx_bait in range(num_bait):
        image_bait = sub_images_bait[idx_bait]
        similarities_this_bait = []
        for idx_sample in range(num_sample):
            image_sample = sub_images_sample[idx_sample]
            similarity = image_bait @ image_sample.t()
            similarities_this_bait.append(similarity.tolist())
        similarities.append(similarities_this_bait)
    similarities = torch.tensor(similarities)
    print(similarities.shape)
    print(similarities[0:5, 0:5])
    return similarities  # (num_bait, num_sample, num_sub_image_bait, num_sub_image_sample)


def summarize_self_similarity(dl_bait, self_similarity):
    images = []
    title = datetime.datetime.now().strftime('%Y%m%d%H%M')
    num = len(dl_bait.dataset)
    for image, label in dl_bait:
        for j in range(len(image)):
            images.append(image[j])

    h = int(math.sqrt(num))
    w = num // h if num % h == 0 else num // h + 1
    print(h, w)
    plot_recovery(images, hw=(h, w), inches=(h * 0.5, w * 0.5), save_path=f'./experiments/test_subimage/{title}.eps')

    with open(f'./experiments/test_subimage/{title}.txt', 'a') as f:
        for idx in range(num):
            self_similarity_this_idx = self_similarity[idx]
            print(f'idx={idx}', file=f)
            for j in range(len(self_similarity_this_idx)):
                print(*([round(s, 4) for s in self_similarity_this_idx[j].tolist()]), sep=",", file=f)
            print('\n\n', file=f)


def summarize_diff_similarity_vanilla(diff_similarity, is_print=True):
    num_bait, num_sample, num_subimage_per_bait, num_subimage_per_sample = diff_similarity.shape
    topk = 4
    diff_similarity_topk_all = []
    for j in range(num_bait):
        diff_similarity_this_image = []
        for k in range(num_subimage_per_bait):
            diff_similarity_this_sub_bait = diff_similarity[j, :, k, :].reshape(-1)
            indices = torch.topk(diff_similarity_this_sub_bait, topk).indices
            topk_this_sub_bait = []
            for indice in indices:
                indice = indice.item()
                idx_sample, idx_sub_image = indice // num_subimage_per_sample, indice % num_subimage_per_sample
                topk_this_sub_bait.append((idx_sample, idx_sub_image))
            diff_similarity_this_image.append(topk_this_sub_bait)
        diff_similarity_topk_all.append(diff_similarity_this_image)

    if is_print:
        for j in range(len(diff_similarity_topk_all)):
            print(*diff_similarity_topk_all[j])
            print(' ')
    diff_similarity_topk_all = torch.tensor(diff_similarity_topk_all)
    print('unique:',{len(diff_similarity_topk_all)})
    print(*[len(torch.unique(diff_similarity_topk_all[:, j, 0])) for j in range(4)])
    print()


def select_bait(fts, num=64, window_size=3):
    fts_flat = fts[:, :, 0:window_size, 0:window_size].reshape(fts.shape[0], -1)
    idx = fts_flat.var(dim=-1).topk(num).indices
    return fts[idx]


def main():
    #  allows redundancy for vefifying idea from different points
    # TODO: calculate quantile of sub-images
    # TODO: draw pictures
    args = parse_args()
    # get dataset
    rs = args.rs
    ds_train, ds_test, resolution, _ = load_dataset(args.root, args.dataset, is_normalize=True)
    ds_sample, _ = get_subdataset(ds_train, p=args.sample_subset, random_seed=rs)
    ds_bait, _ = get_subdataset(ds_test, p=args.bait_subset, random_seed=rs)
    dl_sample, dl_bait = get_dataloader(ds0=ds_sample, ds1=ds_bait, batch_size=64, num_workers=2)

    # get model
    encoder = ToyConvEncoder(input_resolution=resolution, out_resolution=args.out_resolution, is_normalize=True)
    encoder.eval()
    # we only consider normalized encoder, the ONLY variable is down-scaling, i.e, the output resolution.

    fts_sample = pass_forward(encoder, dl_sample)
    fts_bait = pass_forward(encoder, dl_bait)
    fts_bait = select_bait(fts_bait, 64, window_size=args.window_size-1)

    sub_images_sample = moving_window_picker(fts_sample, window_size=args.window_size, padding=args.padding, stride=args.stride, is_rearrange=False)
    sub_images_bait = moving_window_picker(fts_bait, window_size=args.window_size, padding=args.padding, stride=args.stride, is_rearrange=False)

    # calculate self-similarity
    # self_similarity = cal_self_similarity(sub_images_bait, is_normalize=args.is_normalize)
    diff_similarity = cal_diff_similarity(sub_images_sample=sub_images_sample, sub_images_bait=sub_images_bait, is_normalize=args.is_normalize)
    print(diff_similarity.shape)

    # summarize_self_similarity(dl_bait, self_similarity=self_similarity)
    summarize_diff_similarity_vanilla(diff_similarity)

    print('FINISH')


if __name__ == '__main__':
    # use training dataset for input, use test set for constructing.
    main()