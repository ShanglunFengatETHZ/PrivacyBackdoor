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

        self.weights = None  # W for forwarding
        self.bias = None  # b for forwarding

        self._stored_weights = None
        self._stored_bias = None

        # parameters for debugging
        self._update_last_step = []  # in this batch, which picture activates this door
        self._stored_hooked_fish = []  # used for storing raw pictures tha should be shown

    def recovery(self, width) -> list:
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
        super(ToyBackdoor, self).__init__(num_input, num_output)

        self.bias_scaling = nn.Parameter(torch.tensor(bias_scaling), requires_grad=False)
        self.activation = getattr(F, activation)  # we consider relu, gelu, tanh

        self._activate_frequency = torch.zeros(self.num_output)  # how many times has this neuron been activated from start
        self._is_mixture = (torch.zeros(self.num_output) > 1.0)  # whether two images activate this bin at a time
        self._total_replica_within_same_batch = 0

        # self._larger_than_zero = lambda x: x > 0.0
        # self._larger_than_m1 = lambda x: x > -1.0

    def forward(self, features):
        # features: num_samples * out_fts
        signal = self.activation(features @ self.weights + self.bias * self.bias_scaling)

        with torch.no_grad():
            """
            if self.activation is F.tanh:
                self._update_last_step = which_images_activate_this_door(signal, self._larger_than_m1)
            else:
                self._update_last_step = which_images_activate_this_door(signal, self._larger_than_zero)
            """
            self._update_last_step = which_images_activate_this_door(signal)

        return signal

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
        good_images = []

        for j in range(len(self._update_last_step)):
            image_this_door = self._update_last_step[j]
            if len(image_this_door) == 1:
                good_images.append(image_this_door[0])
                self._activate_frequency[j] += 1
            elif len(image_this_door) > 1:
                self._is_mixture[j] = True

        good_images_unique = list(set(good_images))
        self._total_replica_within_same_batch += (len(good_images) - len(good_images_unique))
        for j in range(len(good_images_unique)):
            self._stored_hooked_fish.append(inputs[good_images_unique[j]])


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

        signal = self.activation(vip_signals_sampling + signal_images)

        with torch.no_grad():
            self._update_last_step = which_images_activate_this_door(signal)
        return signal

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


class EasyNet(nn.Module):
    def __init__(self, encoder, backdoor, num_classes=10):
        super().__init__()
        assert isinstance(backdoor, Backdoor), 'backdoor should belong to class Backdoor'
        self.encoder = encoder
        self.backdoor = backdoor
        self.ln = nn.Linear(self.backdoor.num_output, num_classes)

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

    def forward(self, images):
        features_raw = self.encoder(images)
        features_cooked = self.backdoor(features_raw)
        logits = self.ln(features_cooked)
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
    fts_encoder = encoder.out_fts

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
