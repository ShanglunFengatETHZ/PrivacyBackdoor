import torch
from data import get_subdataset, load_dataset, get_dataloader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from edit_transformer import TransformerRegistrar, TransformerWrapper, simulate_block
from tools import indices_period_generator
from train import train_model, get_optimizer
import torch.nn as nn
import copy


def close_block(block):
    block.ln_1.weight.data[:] = 0.
    block.ln_1.bias.data[:] = 0.
    block.self_attention.in_proj_weight.data[:] = 0.
    block.self_attention.in_proj_bias.data[:] = 0.
    block.self_attention.out_proj.weight.data[:] = 0.
    block.self_attention.out_proj.bias.data[:] = 0.

    block.ln_2.weight.data[:] = 0.
    block.ln_2.bias.data[:] = 0.
    block.mlp[0].weight.data[:] = 0.
    block.mlp[0].bias.data[:] = -1e4
    block.mlp[3].weight.data[:] = 0.
    block.mlp[3].bias.data[:] = 0.


def half_activate_transformer(start_idx=1):
    model0 = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    model_new = copy.deepcopy(model0)
    model_new.heads = nn.Linear(768, 10)

    indices_zero = indices_period_generator(num_features=768, head=64, start=7, end=12)

    if indices_zero is not None:
        model_new.conv_proj.weight.data[indices_zero] = 0.
        model_new.conv_proj.bias.data[indices_zero] = 0.
    model_new.class_token.data[:, :, indices_zero] = 0.

    layers = ['encoder_layer_' + str(j) for j in range(12)]

    for j in range(start_idx):
        close_block(getattr(model_new.encoder.layers, layers[j]))

    for j in range(12 - start_idx):
        simulate_block(getattr(model_new.encoder.layers, layers[j + start_idx]), getattr(model0.encoder.layers, layers[j]),
                       zero_indices=indices_zero)

    close_block(model_new.encoder.layers.encoder_layer_11)

    model_new.encoder.ln.weight.data[:] = 1.0
    model_new.encoder.ln.bias.data[:] = 0.0
    model_new.heads.weight.data[:, indices_zero] = 0.
    return model_new


def train_half_transformer():
    # ds_path = '../../cifar10'
    ds_path = '/cluster/project/privsec/data'
    tr_ds, test_ds, resolution, classes = load_dataset(ds_path, 'cifar10', is_normalize=True)
    tr_ds, _ = get_subdataset(tr_ds, p=0.5, random_seed=136)
    bait_ds, _ = get_subdataset(test_ds, p=0.2, random_seed=136)
    tr_dl, test_dl = get_dataloader(tr_ds, batch_size=64, num_workers=2, ds1=test_ds)

    model_new = half_activate_transformer(start_idx=2)

    dataloaders = {'train': tr_dl, 'val': test_dl}

    learning_rate = 1e-2
    optimizer = get_optimizer(model_new, learning_rate, heads_factor=1.0, linear_probe=True)
    num_epochs = 2
    device = 'cuda'

    model_trained = train_model(model_new, dataloaders=dataloaders, optimizer=optimizer, num_epochs=num_epochs, device=device, verbose=False, toy_model=False, direct_resize=224, logger=None)
    save_path = './weights/transformer_cut.pth'
    torch.save(model_trained.state_dict(), save_path)


if __name__ == '__main__':
    train_half_transformer()