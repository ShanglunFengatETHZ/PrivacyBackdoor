import torch
from data import get_subdataset, load_dataset, get_dataloader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from edit_transformer import TransformerRegistrar, TransformerWrapper, indices_period_generator
from train import train_model, get_optimizer

def train_transformer():
    ds_path = '../../cifar10'
    tr_ds, test_ds, resolution, classes = load_dataset(ds_path, 'cifar10', is_normalize=True)
    bait_ds, _ = get_subdataset(test_ds, p=0.2)
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=64, num_workers=2, ds1=test_ds)
    bait_dl = get_dataloader(bait_ds, batch_size=64, num_workers=2)

    model = vit_b_32()

    indices_ft = indices_period_generator(num_features=768, head=64, start=0, end=7)
    indices_bkd = indices_period_generator(num_features=768, head=64, start=7, end=8)
    indices_images = indices_period_generator(num_features=768, head=64, start=8, end=12)

    registrar = TransformerRegistrar(5.0)
    classifier = TransformerWrapper(model, is_double=False, classes=classes, registrar=registrar)
    classifier.divide_this_model(indices_ft=indices_ft, indices_bkd=indices_bkd,indices_img=indices_images)

    noise_scaling = 3.0
    noise = noise_scaling * torch.randn(len(indices_images))

    simulate_images = torch.zeros(resolution, resolution)
    simulate_images[8:24, 8:24] = 1.0
    extracted_pixels = (simulate_images > 0.5)

    classifier.set_conv_encoding(noise=noise, conv_encoding_scaling=50.0, extracted_pixels=extracted_pixels, large_constant=1e6)
    classifier.set_bkd(qthres=0.999, bait_scaling=0.5, zeta=100.0, head_constant=1.0)

    encoderblocks = ['encoder_layer_' + str(j) for j in range(1, 11)]
    classifier.initialize(dl_train=tr_dl, dl_bait=bait_dl, passing_mode='close_block', backdoorblock='encoder_layer_0',
                          encoderblocks=encoderblocks, synthesizeblocks='encoder_layer_11')

    dataloaders = {'train': tr_dl, 'val': test_dl}
    learning_rate = 0.001
    optimizer = get_optimizer(model, learning_rate)
    new_classifier = train_model(classifier, )


def reconstruct_from_transformer():
    pass

if __name__ == '__main__':