import torch
from src.model import ToyConvEncoder
from src.tools import conv_weights_generator, reshape_weight_to_sub_image, reshape_a_feature_to_sub_image

from model import LNRegistrar, ConvRegistrar
from tools import which_images_activate_this_door

if __name__ == '__main__':
    images = torch.rand(200, 3, 32, 32)
    encoder = ToyConvEncoder(32, 4, is_normalize=False)

    weights = conv_weights_generator(in_channels=192, out_channels=50, window_size=3, mode='images', encoder=encoder,
                                     images=images, stride=2, padding=1)
    image_recover = reshape_weight_to_sub_image(weights, 3, 24, 24)

    print('recover sub-images from weights')
    idx = 10
    print(f'idx:{idx}')
    if idx % 4 == 0:
        print(torch.sum(torch.ne(image_recover[idx, :, 8:24, 8:24], images[idx // 4, :, 0:16, 0:16])))
    elif idx % 4 == 1:
        print(torch.sum(torch.ne(image_recover[idx, :, 8:24, 0:24], images[idx // 4, :, 0:16, 8:32])))
    elif idx % 4 == 2:
        print(torch.sum(torch.ne(image_recover[idx, :, 0:24, 8:24], images[idx // 4, :, 8:32, 0:16])))
    else:
        print(torch.sum(torch.ne(image_recover[idx], images[idx // 4, :, 8:32, 8:32])))

    encoder0 = ToyConvEncoder(32, 2, is_normalize=False)
    fts = encoder0(images)
    images_recover_from_fts = reshape_a_feature_to_sub_image(fts[:,:,1,1], 3, 16, 16)
    print('recover images from features')
    print(torch.sum(torch.ne(images[:,:,16:,16:], images_recover_from_fts)))

    """
    num_samples, num_output = 64, 10
    num_exist = 10
    idx = torch.multinomial(torch.ones(num_samples * num_output), num_exist)
    idx_h, idx_w = idx // num_output, idx % num_output
    signal = torch.zeros(num_samples, num_output)
    signal[idx_h, idx_w] = 1.0
    print(f'the signal is')
    print(signal)
    print('\n')

    register = LNRegistrar(num_backdoor=num_output)
    register.update(signal)
    print(f'registrar {register.find_image_valid_activate()}')

    print(f'images {which_images_activate_this_door(signal)}')

    register.record_activation_log()
    print(f'activate valid frequency {register.valid_activate_freq}')
    print(f'is_mixture {register.is_mixture}')
    print(f'update this step: {register.print_update_this_step()}')
    """

    """
    num_samples, num_output, num_h, num_w = 64, 10, 2, 2
    num_exist = 15
    idx = torch.multinomial(torch.ones(num_samples * num_output * num_h * num_w), num_exist)
    idx0, idx1 = idx // (num_h * num_w), idx % (num_h * num_w)
    idx_sample, idx_channel = idx0 // num_output, idx % num_output
    idx_h, idx_w = idx1 // num_w, idx1 % num_w

    signal = torch.zeros(num_samples, num_output, num_h, num_w)
    signal[idx_sample, idx_channel, idx_h, idx_w] = 1.0
    print(f'the signal is')
    print(signal)
    print('\n')

    register = ConvRegistrar(num_backdoor=num_output)
    register.update(signal)
    print(f'registrar {register.find_image_valid_activate()}')

    register.record_activation_log()
    print(f'activate valid frequency {register.valid_activate_freq}')
    print(f'is_mixture {register.is_mixture}')
    print(f'update this step: {register.print_update_this_step()}')
    """





