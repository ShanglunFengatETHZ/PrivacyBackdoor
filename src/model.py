import torch
from torch import nn
import torch.nn.functional as F
from tools import weights_generator


class ToyEncoder(nn.Module):
    def __init__(self, downsampling_factor=None, is_normalize=False, scale_constant=1.0):
        super().__init__()
        self.downsampling_factor = downsampling_factor
        self.is_normalize = is_normalize
        self.scale_constant = nn.Parameter(scale_constant, requires_grad=False)

    def forward(self, images):
        if isinstance(self.downsampling_factor, tuple):
            images_preprocessed = F.interpolate(images, scale_factor=self.downsampling_factor, mode='bilinear')
        else:
            images_preprocessed = images

        num_sample = images_preprocessed.shape[0]
        features_raw = images_preprocessed.reshape(num_sample, -1)

        features = features_raw / features_raw.norm(dim=-1, keepdim=True) if self.is_normalize else features_raw
        return features * self.scale_constant


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

    def forward(self, features):
        return F.relu(features @ self.weights + self.bias * self.bias_scaling)

    def castbait_weight(self, weights):
        if weights.shape[0] == self.num_input and weights.shape[1] == self.num_output:
            self.weights = nn.Parameter(weights, requires_grad=True)

    def castbait_bias(self, bias):
        if bias.shape[0] == self.num_output:
            bias_real = bias / self.bias_scaling
            self.bias = nn.Parameter(bias_real, requires_grad=True)


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

