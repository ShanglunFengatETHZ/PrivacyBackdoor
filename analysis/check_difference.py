import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights


def difference(x_a, x_b, indices, sample, channel):
    return torch.norm(x_a[sample, channel,indices] - x_b[sample,channel,indices]) / torch.norm(x_a[sample,channel, indices])


if __name__ == '__main__':
    path_model_bkd = './weights/transformer_test.pth'
    path_model_cut = './weights/transformer_cut.pth'
    sample = 2
    channel = 0

    model_bkd = torch.load(path_model_bkd, map_location=torch.device('cpu'))
    model_cut = vit_b_32()
    model_cut.heads = nn.Linear(768, 10)
    model_cut.load_state_dict(torch.load(path_model_cut))

    # img = model_bkd.registrar.possible_images[0].unsqueeze(dim=0)
    img = torch.stack(model_bkd.registrar.possible_images, dim=0)
    model_bkd = model_bkd
    model_cut = model_cut

    x_bkd = model_bkd.model._process_input(img)
    n = x_bkd.shape[0]
    batch_class_token = model_bkd.model.class_token.expand(n, -1, -1)
    x_bkd = torch.cat([batch_class_token, x_bkd], dim=1)

    x_cut = model_cut._process_input(img)
    n = x_cut.shape[0]
    batch_class_token = model_cut.class_token.expand(n, -1, -1)
    x_cut = torch.cat([batch_class_token, x_cut], dim=1)

    indices = model_bkd.indices_ft
    indices_passing = torch.cat([model_bkd.indices_ft, model_bkd.indices_bkd])
    print(torch.norm(model_cut.class_token[0,0,indices] - model_bkd.model.class_token[0,0,indices]))

    print(difference(x_cut, x_bkd, indices, sample, channel))

    layers = ['encoder_layer_' + str(j) for j in range(12)]

    for layer in layers:
        this_layer_bkd = getattr(model_bkd.model.encoder.layers, layer)
        this_layer_cut = getattr(model_cut.encoder.layers, layer)
        x_bkd0 = x_bkd
        x_cut0 = x_cut
        print(f'the std of cut features before block is {torch.std(x_cut0[:, channel, indices_passing], dim=-1)}')
        print(f'the std of bkd features before block is {torch.std(x_bkd0[:, channel, indices_passing], dim=-1)}')
        x_bkd = this_layer_bkd(x_bkd)
        x_cut = this_layer_cut(x_cut)
        print(f'after {layer}, the difference is {difference(x_cut, x_bkd, indices, sample, channel)}')
        print('\n')
        # print(f'the norm of features is {x_cut[sample, channel, indices].norm()}')

    z_bkd = model_bkd.model.encoder.ln(x_bkd)
    z_cut = model_cut.encoder.ln(x_cut)

    print(f'after LN, the difference is {difference(z_cut, z_bkd, indices, sample, channel)}')

    z_bkd_cp = model_bkd.model.encoder(x_bkd)
    z_cut_cp = model_cut.encoder(x_cut)
    print(f'after LN, the compared difference is {difference(z_cut_cp, z_bkd_cp, indices, sample, channel)}')







