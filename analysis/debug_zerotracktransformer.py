import torch
from data import get_subdataset, load_dataset, get_dataloader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from edit_transformer import TransformerRegistrar, ViTWrapper, indices_period_generator
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


def moment_order2(x):
    return torch.norm(x) / math.sqrt(len(x))


def build_classifier(classes):
    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    registrar = TransformerRegistrar(30.0)

    classifier = ViTWrapper(model0, is_double=True, classes=classes, registrar=registrar)
    classifier.divide_this_model_horizon(indices_ft=indices_ft, indices_bkd=indices_bkd, indices_img=indices_images)

    encoderblocks = ['encoder_layer_' + str(j) for j in range(1, 11)]
    classifier.divide_this_model_vertical(backdoorblock='encoder_layer_0', encoderblocks=encoderblocks,
                                          synthesizeblocks='encoder_layer_11', filterblock='encoder_layer_1')

    noise_scaling = 3.0
    noise = noise_scaling * torch.randn(len(indices_images) // 2)

    simulate_images = torch.zeros(resolution, resolution)
    simulate_images[8:16, 8:24] = 1.0
    extracted_pixels = (simulate_images > 0.5)

    classifier.set_conv_encoding(noise=noise, conv_encoding_scaling=200.0, extracted_pixels=extracted_pixels,
                                 large_constant=1e9)
    classifier.set_bkd(bait_scaling=0.05, zeta=640.0, num_active_bkd=32, head_constant=1.0)

    classifier.zero_track_initialize(dl_train=tr_dl, passing_mode='zero_pass', v_scaling=1.0, zoom=0.001,
                                     shift_constant=20.0, is_zero_matmul=False)
    return classifier


def output_intermediate(img, model, to=None):
    x = model._process_input(img)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    layers = model.encoder.layers
    if to is not None and to >= 1:
        z = layers[:to](x)
    elif to is not None:
        z = x
    else:
        z = layers(x)
    return z


def plot_correlation(x):
    plt.hist(x)
    plt.xlabel('correlation')
    plt.title(None)
    plt.tight_layout()
    # plt.savefig('experiments/results/20230715_transformer_vanilla/x3_correlation_distribution.eps')
    plt.show()


def plot_multi_hist(x, y, label_x, label_y, z=None, label_z=None, narrow=None, xaxis_label=None, save_to=None):
    if narrow is None:
        narrow = []

    if 'x' not in narrow:
        plt.hist(x, alpha=0.3, label=label_x, color='blue')
    else:
        plt.vlines([torch.tensor(x).mean().item()], 0, 10, colors='blue', alpha=0.3, label=label_x)

    if 'y' not in narrow:
        plt.hist(y, alpha=0.3, label=label_y, color='red')
    else:
        plt.vlines([torch.tensor(y).mean().item()], 0, 10, colors='red', alpha=0.3, label=label_y)

    if z is not None:
        if 'z' not in narrow:
            plt.hist(z, alpha=0.3, label=label_z, color='gray')
        else:
            plt.vlines([torch.tensor(z).mean().item()], 0, 10, colors='gray', alpha=0.3, label=label_z)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xaxis_label)
    plt.title(None)
    plt.legend(loc='upper left')
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
    plt.show()


def cal_layer_variance(z, channel, indices_ft, indices_bkd, indices_img, is_moment=False, C=0.0):
    var_ft_lst = []
    var_bkd_lst = []
    var_img_lst = []
    for j in range(len(z)):
        z_sample = z[j, channel]
        ft_ft = z_sample[indices_ft]
        ft_bkd = z_sample[indices_bkd]
        ft_img = z_sample[indices_img] - C
        var_ft = torch.sqrt(ft_ft.var()).item()
        if not is_moment:
            var_bkd = torch.sqrt(ft_bkd.var()).item()
            var_img = torch.sqrt(ft_img.var()).item()
        else:
            var_bkd = moment_order2(ft_bkd).item()
            var_img = moment_order2(ft_img).item()

        var_ft_lst.append(var_ft)
        var_bkd_lst.append(var_bkd)
        var_img_lst.append(var_img)
    return var_ft_lst, var_bkd_lst, var_img_lst


def cal_sample_variance(z, channel, indices, is_moment=False, C=0.0):
    std_lst = []
    for j in range(len(indices)):
        z_sample = z[:, channel, indices[j]] - C
        if not is_moment:
            std_ft = torch.sqrt(z_sample.var()).item()
        else:
            std_ft = moment_order2(z_sample).item()
        std_lst.append(std_ft)
    return std_lst


def cal_correlation(z1, z2, indices, channel=0):
    if z1.dim() == 3:
        z1 = z1[:, channel, :]
    if z2.dim() == 3:
        z2 = z2[:, channel, :]

    correlation_lst = []
    for j in range(len(indices)):
        this_feature_z1 = z1[:, indices[j]]
        this_feature_z2 = z2[:, indices[j]]

        this_feature_z1 = this_feature_z1.detach().numpy()
        this_feature_z2 = this_feature_z2.detach().numpy()

        r = np.corrcoef(this_feature_z1, this_feature_z2)
        correlation_lst.append(r[1, 0])
    return correlation_lst


if __name__ == '__main__':
    ds_path = '../../cifar10'
    tr_ds, test_ds, resolution, classes = load_dataset(ds_path, 'cifar10', is_normalize=True)
    tr_ds, _ = get_subdataset(tr_ds, p=0.5, random_seed=136)
    bait_ds, _ = get_subdataset(test_ds, p=0.2, random_seed=136)
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=128, num_workers=2, ds1=test_ds)

    indices_ft = indices_period_generator(num_features=768, head=64, start=0, end=7)
    indices_bkd = indices_period_generator(num_features=768, head=64, start=7, end=8)
    indices_images = indices_period_generator(num_features=768, head=64, start=8, end=12)

    # build_classifier(classes)

    # vt32_hf = half_activate_transformer(start_idx=3)
    vt32_hf = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    vt32_hf.heads = nn.Linear(768, 10)
    vt32_hf.load_state_dict(torch.load('./weights/transformer_baseline.pth', map_location='cpu'))
    vt32_hf = vt32_hf.double()
    classifier = torch.load('./weights/transformer_backdoor_v4.pth', map_location='cpu')

    img, _ = next(iter(tr_dl))
    img_big = torch.zeros(128, 3, 224, 224)
    img_big[:, :, :32, :32] = img
    img_big = img_big.double()

    """
    plot z correlation 
    
    z_cf = output_intermediate(img_big, model=classifier.model, to=3)
    z_vt = output_intermediate(img_big, model=vt32_hf, to=3)
    corr = cal_correlation(z_cf, z_vt, indices=indices_ft, channel=1)
    print(corr)
    plot_correlation(corr)
    """


    # plot z variance by sample
    z = output_intermediate(img_big, model=classifier.model, to=None)
    var_ft_lst, var_bkd_lst, var_img_lst = cal_layer_variance(z, channel=0, indices_ft=indices_ft, indices_bkd=indices_bkd,
                                                              indices_img=indices_images, is_moment=True, C=1e9)

    print(var_ft_lst)
    print(var_bkd_lst)
    print(var_img_lst)
    plot_multi_hist(x=var_ft_lst, label_x='features', y=var_img_lst, label_y='pixels', z=var_bkd_lst, label_z='backdoors', narrow=['y'],
                   xaxis_label=r'$\sqrt{\mathbb{E}[Z^2]} & \mathrm{Std}(Z)$', save_to='experiments/results/20230715_transformer_vanilla/z_var_dist_v2.eps')


    """
    # plot z variance by feature
    z = output_intermediate(img_big, model=classifier.model, to=None)
    std_lst_ft = cal_sample_variance(z, channel=0, indices=indices_ft, is_moment=False)
    std_lst_bkd = cal_sample_variance(z, channel=0, indices=indices_bkd, is_moment=False)
    std_lst_img = cal_sample_variance(z, channel=0, indices=indices_images, is_moment=False, C=1e9)

    print(std_lst_ft)
    print(std_lst_bkd)
    print(std_lst_img)
    plot_multi_hist(x=std_lst_ft, label_x='features', y=std_lst_img, label_y='pixels', z=std_lst_bkd, label_z='backdoors',
                    xaxis_label=r'$\mathrm{Std}(X)$', save_to=None)
    """














