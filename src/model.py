import torch
from torch import nn
import torch.nn.functional as F
from tools import weights_generator, pass_forward, moving_window_picker
from tools import reshape_weight_to_sub_image
from tools import select_bait_images, conv_weights_generator, dl2tensor, stringify


class MetaEncoder(nn.Module):
    def __init__(self, input_resolution=32, is_normalize=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.is_normalize = is_normalize

    @property
    def out_fts(self):
        blank = torch.empty(1, 3, self.input_resolution, self.input_resolution)
        fts = self.forward(blank)
        fts = fts.squeeze()
        return fts.shape


class ToyEncoder(MetaEncoder):
    def __init__(self, input_resolution=32, downsampling_factor=None,
                 is_normalize=False, scale_constant=1.0):
        super(ToyEncoder, self).__init__(input_resolution=input_resolution, is_normalize=is_normalize)

        if isinstance(downsampling_factor, list) or isinstance(downsampling_factor, tuple):
            assert len(downsampling_factor) == 2, 'invalid dimension for downsampling, only 2(h,w) is allowed'
            self.downsampling_factor = tuple(downsampling_factor)
        elif isinstance(downsampling_factor, float):
            self.downsampling_factor = tuple([downsampling_factor, downsampling_factor])
        else:
            self.downsampling_factor = None

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


class ToyConvEncoder(MetaEncoder):
    def __init__(self, input_resolution=32, out_resolution=8, is_normalize=True):
        # this is only bi-jection
        super(ToyConvEncoder, self).__init__(input_resolution=input_resolution, is_normalize=is_normalize)
        assert input_resolution % out_resolution == 0, 'toy conv encoder only allows divisible out_resolution'

        self.out_resolution = out_resolution

        self.window_size = input_resolution // out_resolution
        self.out_channels = 3 * self.window_size ** 2

    def forward(self, images):
        if self.is_normalize:
            images = images / images.norm(dim=(1, 2, 3), p=2, keepdim=True)
        fts = moving_window_picker(images, window_size=self.window_size, stride=self.window_size,
                                   padding=0, is_skretch=True)
        return fts


class Backdoor(nn.Module):  # backdoor has to have the method recovery for leaking information
    def __init__(self):
        super().__init__()
        self.weights = None  # W for forwarding
        self.bias = None  # b for forwarding

        self._stored_weights = None
        self._stored_bias = None

        # parameters for debugging
        self._stored_hooked_fishes = []  # used for storing raw pictures tha should be shown
        self.registrar = None

    def recovery(self, width=None) -> list:
        pics = []
        return pics

    def show_initial_weights_as_images(self, width=None):
        # show weight as image to understand the relationship between bait and hooked fish
        return []

    def store_hooked_fish(self, inputs):  # used for extracting raw pictures tha should be shown
        pass


class ToyBackdoor(Backdoor):
    # Wx + cb
    def __init__(self, num_input, num_output, bias_scaling=1.0, activation='relu'):
        super(ToyBackdoor, self).__init__()
        self.num_input = num_input
        self.num_output = num_output

        self.bias_scaling = nn.Parameter(torch.tensor(bias_scaling), requires_grad=False)
        self.activation = getattr(F, activation)  # we consider relu, gelu, tanh

        self.registrar = LNRegistrar(num_backdoor=self.num_output)

    def forward(self, features):
        # features: num_samples * out_fts
        signal = features @ self.weights + self.bias * self.bias_scaling

        with torch.no_grad():
            self.registrar.update(signal)

            if self.registrar.is_log:
                self.registrar.record_activation_log()
        return self.activation(signal)

    def castbait_weight(self, weights):
        assert isinstance(weights, torch.Tensor), 'weight W has to be a tensor'
        assert weights.dim() == 2, 'weight W has to have 2 dimension'
        assert weights.shape[0] == self.num_input and weights.shape[1] == self.num_output, 'weight W has to meet w * h'

        self.weights = nn.Parameter(weights, requires_grad=True)
        self._stored_weights = weights.detach().clone()
        self._stored_weights.requires_grad = False  # this is the upper bound for replication

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

    def get_weights_bias(self, weights, bias):
        self.castbait_weight(weights)
        self.castbait_bias(bias)

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

    def show_initial_weights_as_images(self, width=None):
        # for 3 * 32 * 32, and the uniform weight, the scaling should be - 32.0(EX^2)
        images = []
        if width is None:
            width = int(torch.sqrt(torch.tensor(self.num_input) / 3))
        for j in range(self.num_output):
            weight = self._stored_weights[:, j]
            images.append(weight.reshape(3, width, width))
        return images

    def store_hooked_fish(self, inputs):
        idx_images = self.registrar.find_image_valid_activate()
        for idx in idx_images:
            self._stored_hooked_fishes.append(inputs[idx])


class ToyConv(nn.Module):
    # only used for calculating quantile
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                              bias=False)

    def forward(self, fts):
        return self.conv(fts)

    def get_weight(self, weights):
        assert weights.shape[0] == self.out_channels, 'should input correct shape weight'
        assert weights.shape[1] == self.in_channels, 'should input correct shape weight'
        assert weights.shape[2] == self.kernel_size, 'should input correct shape weight'
        self.conv.weight = nn.Parameter(weights, requires_grad=False)


class ToyConvBackdoor(Backdoor):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 bias_scaling=1.0, activation='relu'):
        super(ToyConvBackdoor, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        assert isinstance(stride, int) and isinstance(padding, int), 'stride & padding should be int'
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

        self.bias = None
        self.bias_scaling = nn.Parameter(torch.tensor(bias_scaling), requires_grad=False)

        self.activation = getattr(F, activation)  # we consider relu, gelu, tanh

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=self.stride, padding=self.padding,
                              bias=False)

        self.registrar = ConvRegistrar(num_backdoor=self.out_channels)

    def forward(self, fts):
        out_fts = self.conv(fts) + self.bias_scaling * self.bias  # (num_samples, out_channels, height, width)

        with torch.no_grad():
            self.registrar.update(out_fts)

            if self.registrar.is_log:
                self.registrar.record_activation_log()

        return self.activation(out_fts)

    def castbait_weight(self, weights):
        assert isinstance(weights, torch.Tensor), 'weight W has to be a tensor'
        assert weights.dim() == 4, 'weight W has to have 4 dimension'
        assert weights.shape[0] == self.out_channels and weights.shape[1] == self.in_channels
        assert weights.shape[2] == self.kernel_size

        self.conv.weight = nn.Parameter(weights, requires_grad=True)
        self._stored_weights = weights.detach().clone()
        self._stored_weights.requires_grad = False  # this is the upper bound for replication

    def castbait_bias(self, bias):
        assert isinstance(bias, torch.Tensor), 'bias b has to be a tensor'
        assert bias.dim() == 1 and len(bias) >= self.out_channels, 'bias b has to have same length as out features'
        bias = bias[:self.out_channels]

        bias = bias.reshape(1, self.out_channels, 1, 1)

        bias_real = bias / self.bias_scaling
        bias_real = bias_real.detach().clone()
        self.bias = nn.Parameter(bias_real, requires_grad=True)
        self._stored_bias = bias_real.detach().clone()
        self._stored_bias.requires_grad = False

    def get_weights_bias(self, weights, bias):
        self.castbait_weight(weights)
        self.castbait_bias(bias)

    def cal_sub_image_resolution(self, channels, height, width):
        return int(torch.sqrt(torch.tensor(channels * height * width) / 3))

    def recovery(self, width=None):
        if width is None:
            width = self.cal_sub_image_resolution(self.in_channels, self.kernel_size, self.kernel_size)
            print(f'the width in recovery is {width}')

        new_weights = self.conv.weight.detach().clone()  # out_channels, in_channels, height, width
        new_bias = self.bias.detach().clone()
        scaling = self.bias_scaling.detach().clone()

        weights_delta = new_weights - self._stored_weights
        bias_delta = new_bias - self._stored_bias
        bias_delta = bias_delta.reshape(-1)

        pics = []

        for j in range(self.out_channels):
            weights_var = weights_delta[j]
            bias_var = bias_delta[j]
            if bias_var.norm() > 1e-8:
                pic = weights_var / bias_var * scaling
                pic = reshape_weight_to_sub_image(pic, image_channel=3, image_width=width, image_height=width)
                pics.append(pic.squeeze())
            else:
                pics.append(torch.ones(3, width, width) * 1e-8)

        return pics

    def show_initial_weights_as_images(self, width=None):
        # for 3 * 32 * 32, and the uniform weight, the scaling should be - 32.0(EX^2)
        images = []
        if width is None:
            width = self.cal_sub_image_resolution(self.in_channels, self.kernel_size, self.kernel_size)

        for j in range(self.out_channels):
            weight = self._stored_weights[j]
            sub_image = reshape_weight_to_sub_image(weight, image_channel=3, image_height=width, image_width=width)
            images.append(sub_image.squeeze())
        return images

    def store_hooked_fish(self, inputs):
        idx_images = self.registrar.find_image_valid_activate()
        for idx in idx_images:
            self._stored_hooked_fishes.append(inputs[idx])

    def get_out_fts(self, in_channels, height, width):
        blank = torch.zeros(1, in_channels, height, width)
        out_fts = self.forward(blank)
        return out_fts.squeeze().shape


class Registrar:
    def __init__(self, num_backdoor, umpire=None, is_log=False):
        self.num_backdoor = num_backdoor
        self.signal = None  # (num_backdoors, num_samples, num_height, num_width)
        self.is_activate = None  # (num_backdoors, num_samples, num_height, num_width)

        if umpire is None:
            self.umpire = self.larger_than_zero
        self.is_log = is_log

        self.valid_activate_freq = torch.zeros(self.num_backdoor)
        self.is_mixture = torch.tensor([False] * self.num_backdoor)

    def update(self, signal):
        pass

    def set_params(self, umpire=None, is_log=False):
        self.umpire = umpire
        self.is_log = is_log

    def print_update_this_step(self):
        update_bkd = []
        for j in range(self.num_backdoor):
            is_activate_bkd = self.is_activate[j]
            if self.is_activate.dim() == 2:
                idx_activate_bkd = torch.nonzero(is_activate_bkd).squeeze()
            else:
                idx_activate_bkd = torch.nonzero(is_activate_bkd)

            if idx_activate_bkd.dim() == 2:
                idx_activate_bkd = [tuple(idx.tolist()) for idx in idx_activate_bkd]
            else:
                idx_activate_bkd = idx_activate_bkd.tolist()
                if not isinstance(idx_activate_bkd, list):
                    idx_activate_bkd = [idx_activate_bkd]
            update_bkd.append(idx_activate_bkd)

        return ';'.join(stringify(update_bkd, self.signal[:, 0, :, :].shape.numel()))

    def find_backdoor_activated(self, is_once):
        return torch.tensor([])

    def find_image_valid_activate(self):
        idx_backdoor_valid_activated = self.find_backdoor_activated(is_once=True)
        backdoor_valid_activated = self.is_activate[idx_backdoor_valid_activated]
        valid_activate_position = torch.nonzero(backdoor_valid_activated)
        # assert len(valid_activate_position) == len(idx_backdoor_valid_activated)
        idx_images = list(set(valid_activate_position[:, 1].tolist()))
        return idx_images

    def record_activation_log(self):
        idx_backdoor_activate_once = self.find_backdoor_activated(is_once=True)
        idx_backdoor_activate_mix = self.find_backdoor_activated(is_once=False)

        for idx in idx_backdoor_activate_once:
            self.valid_activate_freq[idx] += 1

        for idx in idx_backdoor_activate_mix:
            self.is_mixture[idx] = True

    def larger_than_zero(self, x):
        return x > 0.0


class LNRegistrar(Registrar):
    def __init__(self, num_backdoor, umpire=None, is_log=False):
        super(LNRegistrar, self).__init__(num_backdoor=num_backdoor, umpire=umpire, is_log=is_log)

    def update(self, signal):
        self.signal = signal.permute(1, 0)
        assert len(self.signal) == self.num_backdoor, 'the input signal does not have correct '
        self.is_activate = self.umpire(self.signal)

    def find_backdoor_activated(self, is_once):
        if is_once:
            idx = torch.nonzero(self.is_activate.sum(dim=-1) == 1).reshape(-1)
        else:
            idx = torch.nonzero(self.is_activate.sum(dim=-1) > 1).reshape(-1)
        return idx


class ConvRegistrar(Registrar):
    def __init__(self, num_backdoor, umpire=None, is_log=False):
        super(ConvRegistrar, self).__init__(num_backdoor=num_backdoor, umpire=umpire, is_log=is_log)

        self.activation_log = []

    def update(self, signal):
        self.signal = signal.permute(1, 0, 2, 3)
        assert len(self.signal) == self.num_backdoor, 'the input signal does not have correct '
        self.is_activate = self.umpire(self.signal)

    def find_backdoor_activated(self, is_once=False):
        if is_once:
            idx = torch.nonzero(self.is_activate.sum(dim=(1, 2, 3)) == 1).reshape(-1)
        else:
            idx = torch.nonzero(self.is_activate.sum(dim=(1, 2, 3)) > 1).reshape(-1)
        return idx

    def fts_activate_this_door(self, idx):
        backdoor_signal = self.signal[idx]
        backdoor_is_activate = self.is_activate[idx]

        idx_images_activate_this_door = torch.nonzero(backdoor_is_activate.sum(dim=(1, 2)) > 0)
        fts_activate_this_door = backdoor_signal[idx_images_activate_this_door]

        bkd_info = {'backdoor': idx, 'signal': fts_activate_this_door}
        return bkd_info

    def record_activation_log(self):
        idx_backdoor_activate_once = self.find_backdoor_activated(is_once=True)
        idx_backdoor_activate_mix = self.find_backdoor_activated(is_once=False)

        activate_once = []
        for idx in idx_backdoor_activate_once:
            bkd_info = self.fts_activate_this_door(idx)
            activate_once.append(bkd_info)
            self.valid_activate_freq[idx] += 1

        activate_mix = []
        for idx in idx_backdoor_activate_mix:
            bkd_info = self.fts_activate_this_door(idx)
            activate_mix.append(bkd_info)
            self.is_mixture[idx] = True

        self.activation_log.append((activate_once, activate_mix))


class Segmentor(nn.Module):
    # standard output should be between [-a, a], and most outputs are close to [-a, a]
    def __init__(self, num_input, num_output, scaling_constant=10.0, is_changeable=False):
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.scaling_constant = nn.Parameter(torch.tensor(scaling_constant), requires_grad=False)
        self.is_changeable = is_changeable

        self.weights = None
        self.bias = None

    def forward(self, features):
        pass

    def get_weights_bias(self, weights, bias):
        assert weights.shape[0] == self.num_input and weights.shape[1] == self.num_output, 'Invalid weights size'
        self.weights = nn.Parameter(weights, requires_grad=self.is_changeable)

        if isinstance(bias, float):
            bias = bias * torch.ones(self.num_output)
        assert len(bias) == self.num_output, 'Invalid bias size'
        self.bias = nn.Parameter(bias, requires_grad=self.is_changeable)

    def set_module(self):
        pass


class TanhSegmentor(Segmentor):
    def __init__(self, num_input, num_output, scaling_constant, is_changeable=False):
        super(TanhSegmentor, self).__init__(num_input, num_output, scaling_constant=scaling_constant, is_changeable=is_changeable)

    def forward(self, features):
        signals = features @ self.weights + self.bias
        return F.tanh(self.scaling_constant * signals)


class TwinTrackBackdoor(ToyBackdoor):
    def __init__(self, num_input, num_output, bias_scaling=1.0, activation='relu',
                 segmentor=None, is_seg2bkd_native=True, is_seg2bkd_fixed=True):
        super(TwinTrackBackdoor, self).__init__(num_input, num_output, bias_scaling=bias_scaling, activation=activation)

        # NOW we only consider the number of vip signals meets the number of backdoors
        self.segmentor = segmentor

        self.is_seg2bkd_native = is_seg2bkd_native
        self.is_seg2bkd_fixed = is_seg2bkd_fixed

        self.num_vip_signals = None
        self.seg2bkd_weights, self.seg2bkd_bias = None, None

    def forward(self, features):
        vip_signals = self.segmentor(features)

        if self.is_seg2bkd_native:
            vip_signals_sampling = self.ln(vip_signals)
        else:
            vip_signals_sampling = None

        signal_images = features @ self.weights + self.bias * self.bias_scaling
        signal = vip_signals_sampling + signal_images

        with torch.no_grad():
            self.registrar.update(signal)

            if self.registrar.is_log:
                self.registrar.record_activation_log()

        return self.activation(signal)

    def ln(self, vip_signals):
        return self.seg2bkd_weights * vip_signals + self.seg2bkd_bias

    def set_segmentor_params(self, segmentor=None, is_seg2bkd_native=True):
        self.segmentor = segmentor
        self.is_seg2bkd_native = is_seg2bkd_native
        self.num_vip_signals = segmentor.num_output
        if self.is_seg2bkd_native:
            assert self.num_vip_signals == self.num_output, 'NATIVE mode: num vip signals should be in line with num output'''

    def get_seg2bkd_weights_bias(self, weights, bias):
        if self.is_seg2bkd_native:
            self.seg2bkd_weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
            self.seg2bkd_bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
        elif self.is_seg2bkd_fixed:
            assert isinstance(weights, torch.Tensor) and isinstance(bias, torch.Tensor), 'should input tenor'
            assert weights.shape[0] == self.num_vip_signals and weights.shape[1] == self.num_output, 'weights between segmentor & backdoor should meet dimension'
            self.seg2bkd_weights = nn.Parameter(weights, requires_grad=False)
            self.seg2bkd_bias = nn.Parameter(bias, requires_grad=False)


class MetaNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.ln = None

    def set_ln_weights(self, constant=1.0):  # for this toy EasyNet, I only use sparse weight
        in_features, out_features = self.ln.in_features, self.ln.out_features
        A = weights_generator(in_features, out_features, constant=constant, mode='fixed_sparse', is_normalize=False)
        self.ln.weight.data = A.transpose(0, 1)

    def set_ln_bias(self, b=0.0):
        num_classes = len(self.ln.bias)
        self.ln.bias.data = torch.ones(num_classes) * b

    def get_ln_weights_bias(self, constant, b):
        self.set_ln_weights(constant)
        self.set_ln_bias(b)


class EasyNet(MetaNet):
    def __init__(self, encoder, backdoor, num_classes=10):
        super(EasyNet, self).__init__(num_classes=num_classes)
        assert isinstance(encoder, ToyEncoder), 'encoder should be ToyEncoder for EasyNet'
        assert isinstance(backdoor, Backdoor), 'backdoor should belong to class Backdoor'
        self.encoder = encoder
        self.backdoor = backdoor
        self.ln = nn.Linear(self.backdoor.num_output, num_classes)

    def forward(self, images):
        features_raw = self.encoder(images)
        features_cooked = self.backdoor(features_raw)
        logits = self.ln(features_cooked)
        return logits


class ConvNet(MetaNet):
    def __init__(self, encoder, backdoor, num_classes=10, use_pool=False):
        super(ConvNet, self).__init__(num_classes=num_classes)
        assert isinstance(encoder, ToyConvEncoder), 'encoder of ConvNet should belong to class ToyConvEncoder'
        assert isinstance(backdoor, ToyConvBackdoor), 'backdoor of ConvNet should belong to class ToyConvBackdoor'
        self.encoder = encoder
        self.backdoor = backdoor
        self.use_pool = use_pool

        encoder_output_shape = self.encoder.out_fts
        assert encoder.out_fts[0] == backdoor.in_channels, 'the out channels of encoder should be the same as the in channels of backdoor'
        backdoor_output_shape = self.backdoor.get_out_fts(*encoder_output_shape)

        if self.use_pool:
            num_features = backdoor_output_shape[0]
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            num_features = backdoor_output_shape.numel()
            self.pool = None
        self.ln = nn.Linear(num_features, num_classes)

    def forward(self, images):
        features_raw = self.encoder(images)
        features_cooked = self.backdoor(features_raw)
        if self.use_pool:
            features_pooled = self.pool(features_cooked)
        else:
            features_pooled = features_cooked
        features = torch.flatten(features_pooled, start_dim=1)
        logits = self.ln(features)
        return logits


def set_weights_bias(module,
                     weight_mode='gaussian', bias_mode='constant', weights_details=None,  bias_details=None,
                     encoder=None, is_seg_bkd=False):
    # weights details(quantile) : {is_normalize, constant}
    # weights_details(images) : {is_normalize, constant, dl_bait_images}
    # bias_details(quantile) : {quantile, dl_target_distribution}
    # bias details(constant) : {constant}
    if weight_mode == 'images':
        dl_bait_images = weights_details.pop('dl_bait_images')
        backdoor_fts = pass_forward(encoder, dl_bait_images)
        weights_details['image_fts'] = backdoor_fts

    weights = weights_generator(module.num_input, module.num_output, mode=weight_mode, **weights_details)

    if bias_mode == 'quantile' and 0.0 <= bias_details['quantile'] <= 1.0:
        fts_target_distribution = pass_forward(encoder, bias_details['dl_target_distribution'])
        output_target_distribution = fts_target_distribution @ weights

        bias = - 1.0 * torch.tensor([torch.quantile(output_target_distribution[:, j], q=bias_details['quantile'], keepdim=False, interpolation='linear') for j in range(output_target_distribution.shape[1])])
        print('We choose bias:', *bias.tolist())
    else:
        bias = torch.ones(module.num_output) * bias_details['constant']

    if is_seg_bkd:
        module.get_seg2bkd_weights_bias(weights, bias)
    else:
        module.get_weights_bias(weights, bias)


def instantiate_a_twin_track_backdoor(backdoor, num_vip_signals, segmentor_type='tanh', segmentor_scaling_constant=10.0,
                                      is_seg2bkd_native=True, seg2bkd_details=None,
                                      seg_weight_mode='constant', seg_weight_details=None, seg_bias_mode='constant', seg_bias_details=None,
                                      encoder=None):
    # seg2bkd_details(native) = {coeff, b}
    # seg2bkd_details(else) = {weight_mode, bias_mode, weights_details, bias_details}

    assert isinstance(backdoor, TwinTrackBackdoor), 'backdoor should be TwinTrackBackdoor'

    # set segmentor
    num_input = backdoor.num_input  # currently, we only consider the input of segmentor is the same as backdoor
    if segmentor_type == 'tanh':
        segmentor = TanhSegmentor(num_input=num_input, num_output=num_vip_signals,  #  NOW we only consider unchangebale module
                                  scaling_constant=segmentor_scaling_constant, is_changeable=False)
        set_weights_bias(segmentor, weight_mode=seg_weight_mode, bias_mode=seg_bias_mode,
                         weights_details=seg_weight_details, bias_details=seg_bias_details, encoder=encoder)
    else:
        segmentor = None

    # set seg2bkd
    backdoor.set_segmentor_params(segmentor=segmentor, is_seg2bkd_native=is_seg2bkd_native)
    if backdoor.is_seg2bkd_native:
        seg2bkd_details = {} if seg2bkd_details is None else seg2bkd_details
        backdoor.get_seg2bkd_weights_bias(seg2bkd_details.get('coeff', 1.0), seg2bkd_details.get('b', 0.0))
    else:
        set_weights_bias(backdoor, weight_mode=seg2bkd_details['weight_mode'], bias_mode=seg2bkd_details['bias_mode'],
                         weights_details=seg2bkd_details['weights_details'],
                         bias_details=seg2bkd_details['bias_details'], encoder=encoder, is_seg_bkd=True)


def make_an_toy_net(input_resolution=32, num_class=10,  # background
                    encoder_details=None,  # encoder
                    num_leaker=64, bias_scaling=1.0, activation='relu', use_twin_track_backdoor=False,  # backdoor meta
                    bkd_weight_mode='uniform', bkd_weight_details=None, bkd_bias_mode='constant', bkd_bias_details=None,
                    twin_track_backdoor_details=None,  # backdoor parameter
                    ln_details=None):  # linear layer
    # encoder_details:{downsampling_factor, is_normalize, scale_constant}
    # ln_details: {constant, b}
    # twin_track_backdoor_details = {segmentor_type, segmentor_scaling_constant, is_seg2bkd_native, seg2bkd_details, seg_weight_mode, seg_weight_details, seg_bias_mode, seg_bias_details}

    encoder = ToyEncoder(input_resolution=input_resolution, **encoder_details)
    fts_encoder = encoder.out_fts[0]

    if use_twin_track_backdoor and isinstance(twin_track_backdoor_details, dict):
        backdoor = TwinTrackBackdoor(num_input=fts_encoder, num_output=num_leaker, bias_scaling=bias_scaling, activation=activation)
        instantiate_a_twin_track_backdoor(backdoor, num_vip_signals=backdoor.num_output, encoder=encoder, **twin_track_backdoor_details)
    else:
        backdoor = ToyBackdoor(num_input=fts_encoder, num_output=num_leaker, bias_scaling=bias_scaling, activation=activation)

    set_weights_bias(backdoor, weight_mode=bkd_weight_mode, bias_mode=bkd_bias_mode,
                     weights_details=bkd_weight_details, bias_details=bkd_bias_details, encoder=encoder)

    toy_net = EasyNet(encoder, backdoor, num_class)
    toy_net.get_ln_weights_bias(**ln_details)

    return toy_net


def make_conv_net(input_resolution=32, num_classes=10,
                  encoder_details=None, backdoor_arch_details=None, ln_details=None,
                  backdoor_weight_mode='gassuian', backdoor_weight_details=None, backdoor_bias_mode='quantile', backdoor_bias_details=None,
                  num_leaker=64, bias_scaling=1.0, activation='relu', use_pool=False):

    encoder = ToyConvEncoder(input_resolution, **encoder_details)
    backdoor = ToyConvBackdoor(out_channels=num_leaker, bias_scaling=bias_scaling, activation=activation, **backdoor_arch_details)
    # backdoor arch details: in_channels, kernel_size, stride, padding

    window_size = backdoor_weight_details['window_size']
    is_normalize = backdoor_weight_details.get('is_normalize', True)
    constant = backdoor_weight_details.get('constant', 1.0)
    # backdoor weight details: window_size, is_normalize, constant, images_details:{padding, stride, dl_bait_images}
    if backdoor_weight_mode == 'images':
        images_details = backdoor_weight_details.get('images_details', {})
        stride, padding = images_details.get('stride', 1), images_details.get('padding', 0)
        mode, dl_images_bait = images_details.get('mode', None), images_details['dl_bait_images']
        images_bait, = dl2tensor(dl_images_bait)
        print(f'mode for extracting image bait is {mode}')
        selected_images = select_bait_images(images=images_bait, num_selected=num_leaker, mode=mode)

        weights = conv_weights_generator(in_channels=encoder.out_channels, out_channels=backdoor.out_channels, window_size=window_size,
                                         mode=backdoor_weight_mode, is_normalize=is_normalize, constant=constant,
                                         encoder=encoder, images=selected_images, stride=stride, padding=padding)
    else:  # gaussian
        weights = conv_weights_generator(in_channels=encoder.out_channels, out_channels=backdoor.out_channels,
                                         window_size=window_size, is_normalize=is_normalize, constant=constant, mode=backdoor_weight_mode)

    if backdoor_bias_mode == 'quantile':
        conv_vanilla = ToyConv(out_channels=num_leaker, **backdoor_arch_details)
        conv_vanilla.get_weight(weights)
        fts_tgt_distri = pass_forward(nn.Sequential(encoder, conv_vanilla), backdoor_bias_details['dl_target_distribution'])
        output_target_distribution = fts_tgt_distri.permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0)
        print(f'number of output target distribution {output_target_distribution.shape[0]}, number of features {output_target_distribution.shape[1]}')

        bias = - 1.0 * torch.tensor([torch.quantile(
            output_target_distribution[:, j], q=backdoor_bias_details['quantile'], keepdim=False, interpolation='linear')
            for j in range(output_target_distribution.shape[1])])
        print('We choose bias:', *bias.tolist())
    else:
        bias = None
    backdoor.get_weights_bias(weights=weights, bias=bias.reshape(-1))

    toy_conv_net = ConvNet(encoder, backdoor, num_classes=num_classes, use_pool=use_pool)
    toy_conv_net.get_ln_weights_bias(**ln_details)

    return toy_conv_net



