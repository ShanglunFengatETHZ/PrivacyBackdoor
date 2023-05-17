import torch
from torch import nn
import torch.nn.functional as F
from tools import weights_generator, which_images_activate_this_door, pass_forward


class ToyEncoder(nn.Module):
    def __init__(self, input_resolution=32, downsampling_factor=None,
                 is_normalize=False, scale_constant=1.0):
        super().__init__()
        self.input_resolution = input_resolution

        if isinstance(downsampling_factor, list) or isinstance(downsampling_factor, tuple):
            assert len(downsampling_factor) == 2, 'invalid dimension for downsampling, only 2(h,w) is allowed'
            self.downsampling_factor = tuple(downsampling_factor)
        elif isinstance(downsampling_factor, float):
            self.downsampling_factor = tuple([downsampling_factor, downsampling_factor])
        else:
            self.downsampling_factor = None

        self.is_normalize = is_normalize
        self.scale_constant = scale_constant

    def forward(self, images):
        if isinstance(self.downsampling_factor, tuple):
            images_preprocessed = F.interpolate(images, scale_factor=self.downsampling_factor, mode='bilinear')
        else:
            images_preprocessed = images

        num_sample = images_preprocessed.shape[0]
        features_raw = images_preprocessed.reshape(num_sample, -1)

        if self.is_normalize:
            features = features_raw / features_raw.norm(dim=-1, keepdim=True)
        else:
            features = features_raw
        return features * self.scale_constant

    @property
    def out_fts(self):
        blank = torch.empty(1, 3, self.input_resolution, self.input_resolution)
        if isinstance(self.downsampling_factor, tuple):
            blank_rescale = F.interpolate(blank, scale_factor=self.downsampling_factor, mode='bilinear')
            return blank_rescale.shape.numel()
        else:
            return blank.shape.numel()


class Backdoor(nn.Module):  # backdoor has to have the method recovery for leaking information
    def __init__(self, num_input, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output

    def recovery(self, width) -> list:
        pics = []
        return pics


class ToyBackdoor(Backdoor):
    # Wx + cb
    def __init__(self, num_input, num_output, bias_scaling=1.0):
        # TODO: can we store the good pictures?
        super(ToyBackdoor, self).__init__(num_input, num_output)

        self.bias_scaling = nn.Parameter(torch.tensor(bias_scaling), requires_grad=False)

        self.weights = None  # W for forwarding
        self.bias = None  # b for forwarding

        self._stored_weights = None
        self._stored_bias = None

        self._update_last_step = []

        self._stored_hooked_fish = []

    def forward(self, features):
        # features: num_samples * out_fts
        signal = F.relu(features @ self.weights + self.bias * self.bias_scaling)

        with torch.no_grad():
            self._update_last_step = which_images_activate_this_door(signal)
        return signal

    def castbait_weight(self, weights):
        assert isinstance(weights, torch.Tensor), 'weight W has to be a tensor'
        assert weights.dim() == 2, 'weight W has to have 2 dimension'
        assert weights.shape[0] == self.num_input and weights.shape[1] == self.num_output, 'weight W has to meet w * h'

        self.weights = nn.Parameter(weights, requires_grad=True)
        self._stored_weights = weights.detach().clone()
        self._stored_weights.requires_grad = False

    def castbait_bias(self, bias):
        assert isinstance(bias, torch.Tensor), 'bias b has to be a tensor'
        assert len(bias) == self.num_output, 'bias b has to have same length as out features'

        if bias.dim() == 1:
            bias = bias.reshape(1, -1)

        bias_real = bias / self.bias_scaling
        bias_real = bias_real.detach().clone()
        self.bias = nn.Parameter(bias_real, requires_grad=True)
        self._stored_bias = bias_real.detach().clone()
        self._stored_bias.requires_grad = False

    def recovery(self, width=None):
        if width is None:
            width = int(torch.sqrt(torch.tensor(self.num_input) / 3))

        new_weights = self.weights.detach().clone()
        new_bias = self.bias.detach().clone()
        scaling = self.bias_scaling.detach().clone()

        weights_delta = new_weights - self._stored_weights
        bias_delta = new_bias - self._stored_bias

        pics = []
        for j in range(self.num_output):
            weights_var = weights_delta[:, j]
            bias_var = bias_delta[0, j]
            if bias_var.norm() > 1e-8:
                pic = weights_var / bias_var * scaling
                pic = pic.reshape(3, width, width)
                pics.append(pic)
            else:
                pics.append(torch.ones(3, width, width) * 1e-8)
        return pics

    def show_weights(self, width=None):
        # for 3 * 32 * 32, and the uniform weight, the scaling should be - 32.0(EX^2)
        images = []
        if width is None:
            width = int(torch.sqrt(torch.tensor(self.num_input) / 3))
        for j in range(self.num_output):
            weight = self._stored_weights[:, j]
            images.append(weight.reshape(3, width, width))
        return images

    def store_hooked_fish(self, inputs):
        good_images = []
        for image_this_door in self._update_last_step:
            if len(image_this_door) == 1:
                good_images.append(image_this_door[0])
        good_images_unique = list(set(good_images))
        for j in range(len(good_images_unique)):
            self._stored_hooked_fish.append(inputs[good_images_unique[j]])


class EasyNet(nn.Module):
    def __init__(self, encoder, backdoor, num_classes=10):
        super().__init__()
        assert isinstance(backdoor, Backdoor), 'backdoor should belong to class Backdoor'
        self.encoder = encoder
        self.backdoor = backdoor
        self.ln = nn.Linear(self.backdoor.num_output, num_classes)

    def _set_weights(self, constant=1.0):  # for this toy EasyNet, I only use sparse weight
        in_features, out_features = self.ln.in_features, self.ln.out_features
        A = weights_generator(in_features, out_features, constant=constant, mode='fixed_sparse', is_normalize=False)
        self.ln.weight.data = A.transpose(0, 1)

    def _set_bias(self, b=0.0):
        num_classes = len(self.ln.bias)
        self.ln.bias.data = torch.ones(num_classes) * b

    def forward(self, images):
        features_raw = self.encoder(images)
        features_cooked = self.backdoor(features_raw)
        logits = self.ln(features_cooked)
        return logits


def make_an_toy_net(input_resolution=32, num_class=10,
                    downsampling_factor=None, is_encoder_normalize=False, encoder_scale_constant=1.0,
                    num_leaker=64, bias_scaling=1.0, backdoor_weight_mode='uniform', is_backdoor_normalize=True, dl_backdoor_images=None, backdoor_bias=-1.0, backdoor_weight_factor=1.0,
                    ln_weight_factor=1.0, ln_bias_factor=0.0):

    encoder = ToyEncoder(input_resolution=input_resolution, downsampling_factor=downsampling_factor, is_normalize=is_encoder_normalize, scale_constant=encoder_scale_constant)
    fts_encoder = encoder.out_fts

    backdoor = ToyBackdoor(num_input=fts_encoder, num_output=num_leaker, bias_scaling=bias_scaling)
    if dl_backdoor_images is not None and backdoor_weight_mode == 'images':
        backdoor_fts = pass_forward(encoder, dl_backdoor_images)
    else:
        backdoor_fts = None
    backdoor_weight = weights_generator(backdoor.num_input, backdoor.num_output, mode=backdoor_weight_mode, is_normalize=is_backdoor_normalize, image_fts=backdoor_fts, constant=backdoor_weight_factor)
    backdoor.castbait_weight(backdoor_weight)
    backdoor_bias = torch.ones(backdoor.num_output) * backdoor_bias
    backdoor.castbait_bias(backdoor_bias)

    toy_net = EasyNet(encoder, backdoor, num_class)
    toy_net._set_weights(constant=ln_weight_factor)
    toy_net._set_bias(b=ln_bias_factor)
    return toy_net

