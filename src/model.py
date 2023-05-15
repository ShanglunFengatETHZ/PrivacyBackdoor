import torch
from torch import nn
import torch.nn.functional as F
from tools import weights_generator


class ToyEncoder(nn.Module):
    # TDOD: attribute what is the output
    def __init__(self, downsampling_factor=None, is_normalize=False, scale_constant=1.0, input_resolution=32):
        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.is_normalize = is_normalize
        self.scale_constant = nn.Parameter(scale_constant, requires_grad=False)
        self.input_resolution = input_resolution
        self.out_fts = self.calculate_out_fts()

    def forward(self, images):
        if isinstance(self.downsampling_factor, tuple):
            images_preprocessed = F.interpolate(images, scale_factor=self.downsampling_factor, mode='bilinear')
        else:
            images_preprocessed = images

        num_sample = images_preprocessed.shape[0]
        features_raw = images_preprocessed.reshape(num_sample, -1)

        features = features_raw / features_raw.norm(dim=-1, keepdim=True) if self.is_normalize else features_raw
        return features * self.scale_constant

    def calculate_out_fts(self):
        blank = torch.empty(1, 3, self.input_resolution, self.input_resolution)
        blank_rescale = F.interpolate(blank, scale_factor=self.downsampling_factor, mode='bilinear')
        return blank_rescale.shape.numel()


class Backdoor(nn.Module):
    # use abstract class
    def __init__(self):
        super().__init__()


class ToyBackdoor(Backdoor):
    def __init__(self, num_input, num_output, bias_scaling=1.0):
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.bias_scaling = nn.Parameter(bias_scaling, requires_grad=False)
        self.weights = None
        self.bias = None

        self._stored_weights = None
        self._stored_bias = None

    def forward(self, features):
        return F.relu(features @ self.weights + self.bias * self.bias_scaling)

    def castbait_weight(self, weights):
        if weights.shape[0] == self.num_input and weights.shape[1] == self.num_output:
            self.weights = nn.Parameter(weights, requires_grad=True)
            self._stored_weights = weights.clone()
            self._stored_weights.requires_grad = False

    def castbait_bias(self, bias):
        if bias.shape[0] == self.num_output:
            bias_real = bias / self.bias_scaling
            self.bias = nn.Parameter(bias_real, requires_grad=True)
            self._stored_bias = bias.clone()
            self._stored_bias.requires_grad = False

    def recovery(self, width):
        device = self._stored_weights.device
        new_weights = self.weights.detach().clone().to(device)
        new_bias = self.bias.detach().clone().to(device)

        weights_delta = new_weights - self._stored_weights
        bias_delta = new_bias - self._stored_bias

        pics = []
        for j in len(bias_delta):
            weights_var = weights_delta[j]
            bias_var = bias_delta[j]
            if new_bias.norm() > 1e-6:
                scaling = self.bias_scaling.to(device)
                pic = weights_var / bias_var * scaling
                pic.reshape(3, width, width)
                pics.append(pic)
            else:
                pics.append(torch.ones(3, width, width) * 1e-6)
        return pics


class EasyNet(nn.Module):
    def __init__(self, encoder, backdoor, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.backdoor = backdoor
        self.ln = nn.Linear(self.backdoor.num_output, num_classes)
        self.set_weights()
        self.set_bias()

    def set_weights(self, mode='fixed_sparse', c=1.0):
        in_features, out_features = self.ln.in_features, self.ln.out_features
        self.ln.weight.data = weights_generator(in_features, out_features, mode=mode, is_normalize=False, c=c)

    def set_bias(self, mode='constant', b=0.0):
        if mode == 'constant':
            num_classes = len(self.ln.bias)
            self.ln.bias.data = torch.ones(num_classes) * b

    def forward(self, images):
        features = self.encoder(images)
        features_dirty = self.backdoor(features)
        logit = self.ln(features_dirty)
        return logit


def make_an_toy_net(input_resolution=32, num_class=10,
                    downsampling_factor=None, is_encoder_normalize=False, encoder_scale_constant=1.0,
                    num_leaker=64, bias_scaling=1.0, backdoor_weight_mode='uniform', is_backdoor_normalize=True, backdoor_images=None, backdoor_bias=-1.0,
                    ln_weight_mode='fix_sparse', ln_weight_factor=1.0, ln_bias_mode='constant', ln_bias_factor=0.0):
    encoder = ToyEncoder(downsampling_factor=downsampling_factor, is_normalize=is_encoder_normalize, scale_constant=encoder_scale_constant, input_resolution=input_resolution)
    fts_encoder = encoder.out_fts
    backdoor = ToyBackdoor(num_input=fts_encoder, num_output=num_leaker, bias_scaling=bias_scaling)
    backdoor_fts = encoder(backdoor_images)
    backdoor_weight = weights_generator(backdoor.num_input, backdoor.num_output, mode=backdoor_weight_mode, is_normalize=is_backdoor_normalize, images=backdoor_fts)
    backdoor.castbait_weight(backdoor_weight)
    backdoor_bias = torch.ones(backdoor.num_output) * backdoor_bias
    backdoor.castbait_bias(backdoor_bias)

    toy_net = EasyNet(encoder, backdoor, num_class)
    toy_net.set_weights(mode=ln_weight_mode, c=ln_weight_factor)
    toy_net.set_bias(mode=ln_bias_mode, b=ln_bias_factor)
    return toy_net

