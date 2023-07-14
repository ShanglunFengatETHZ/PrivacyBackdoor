import torch
from data import get_subdataset, load_dataset, get_dataloader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from edit_transformer import TransformerRegistrar, TransformerWrapper, indices_period_generator
from train import train_model, get_optimizer


def train_transformer():
    # ds_path = '../../cifar10'
    ds_path = '/cluster/project/privsec/data'
    tr_ds, test_ds, resolution, classes = load_dataset(ds_path, 'cifar10', is_normalize=True)
    tr_ds, _ = get_subdataset(tr_ds, p=0.5, random_seed=136)
    bait_ds, _ = get_subdataset(test_ds, p=0.2, random_seed=136)
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=64, num_workers=2, ds1=test_ds)

    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)

    indices_ft = indices_period_generator(num_features=768, head=64, start=0, end=7)
    indices_bkd = indices_period_generator(num_features=768, head=64, start=7, end=8)
    indices_images = indices_period_generator(num_features=768, head=64, start=8, end=12)

    registrar = TransformerRegistrar(1000.0)

    classifier = TransformerWrapper(model0, is_double=True, classes=classes, registrar=registrar)
    classifier.divide_this_model_horizon(indices_ft=indices_ft, indices_bkd=indices_bkd, indices_img=indices_images)

    """
    encoderblocks = ['encoder_layer_' + str(j) for j in range(1, 11)]
    classifier.divide_this_model_vertical(backdoorblock='encoder_layer_0', encoderblocks=encoderblocks, synthesizeblocks='encoder_layer_11')

    noise_scaling = 3.0
    noise = noise_scaling * torch.randn(len(indices_images))

    simulate_images = torch.zeros(resolution, resolution)
    simulate_images[8:24, 8:24] = 1.0
    extracted_pixels = (simulate_images > 0.5)

    classifier.set_conv_encoding(noise=noise, conv_encoding_scaling=20.0, extracted_pixels=extracted_pixels, large_constant=1e4)
    classifier.set_bkd(bait_scaling=0.5, zeta=640.0, num_active_bkd=32, head_constant=1.0)

    classifier.initialize(dl_train=tr_dl, passing_mode='half_divided', v_scaling=1.0, is_zero_matmul=False)

    dataloaders = {'train': tr_dl, 'val': test_dl}
    learning_rate = 0.0001
    optimizer = get_optimizer(classifier.model, learning_rate, heads_factor=100.0)
    num_epochs = 5
    device = 'cuda'
    """

    encoderblocks = ['encoder_layer_' + str(j) for j in range(1, 11)]
    classifier.divide_this_model_vertical(backdoorblock='encoder_layer_0', encoderblocks=encoderblocks, synthesizeblocks='encoder_layer_11', zeroout='encoder_layer_1', filterblock='encoder_layer_2')

    noise_scaling = 3.0
    noise = noise_scaling * torch.randn(len(indices_images) // 2)

    simulate_images = torch.zeros(resolution, resolution)
    simulate_images[8:16, 8:24] = 1.0
    extracted_pixels = (simulate_images > 0.5)

    classifier.set_conv_encoding(noise=noise, conv_encoding_scaling=200.0, extracted_pixels=extracted_pixels,
                                 large_constant=1e9)
    classifier.set_bkd(bait_scaling=0.05, zeta=64000.0, num_active_bkd=32, head_constant=1.0)

    classifier.zero_track_initialize(dl_train=tr_dl, passing_mode='zero_pass', v_scaling=1.0, is_zero_matmul=False)

    dataloaders = {'train': tr_dl, 'val': test_dl}
    learning_rate = 1e-8
    optimizer = get_optimizer(classifier.model, learning_rate, heads_factor=5e5)
    num_epochs = 1
    device = 'cuda'
    save_path = './weights/transformer_test.pth'

    new_classifier = train_model(classifier, dataloaders=dataloaders, optimizer=optimizer, num_epochs=num_epochs, device=device, verbose=False, toy_model=False, direct_resize=224, logger=None)
    torch.save(new_classifier, save_path)


if __name__ == '__main__':
    train_transformer()