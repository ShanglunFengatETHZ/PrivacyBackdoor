import torch
import torch.nn as nn
from tools import cal_set_difference_seq, weights_generator
import random
from collections import Counter
from types import MethodType


class NativeMLP(nn.Module):
    def __init__(self, hidden_size, input_size, classes, activation='ReLU', preprocess_information=None):
        super().__init__()
        self.preprocess_information = preprocess_information
        random_image = torch.rand(input_size)
        random_input = self._preprocess(random_image.unsqueeze(dim=0))
        input_entries = random_input.shape[1]
        assert len(hidden_size) == 2, f'only support hidden size 2, but get {hidden_size}'
        self.input_layer = nn.Linear(input_entries, hidden_size[0])
        self.act01 = getattr(nn, activation)()
        self.intermediate_layer = nn.Linear(hidden_size[0], hidden_size[1])
        self.act12 = getattr(nn, activation)()
        self.output_layer = nn.Linear(hidden_size[1], classes)

        self.num_backdoors = 0
        self.init_input_layer_weight = self.input_layer.weight.detach().clone()
        self.init_input_layer_bias = self.input_layer.bias.detach().clone()
        self.possible_images = None
        self.arch = {'hidden_size': hidden_size, 'input_size': input_size, 'classes': classes, 'activation': activation,
                     'preprocess_information':preprocess_information}

    def forward(self, images):
        inputs = self._preprocess(images)
        x = self.input_layer(inputs)
        x = self.act01(x)
        self._register(images, x)
        x = self.intermediate_layer(x)
        x = self.act12(x)
        output = self.output_layer(x)

        return output

    def _preprocess(self, images):
        x = native_preprocess(images, self.preprocess_information)
        return x

    def _register(self, images, x):
        images = images.detach().clone()
        act = x.detach().clone()[:, :self.num_backdoors]
        is_activated = torch.gt(act, 1e-5)
        indices = torch.nonzero(is_activated)
        idx_sample, idx_backdoor = indices[:, 0], indices[:, 1]
        for j in range(len(idx_backdoor)):
            self.possible_images[idx_backdoor[j]].append({'image':images[idx_sample[j]], 'act':act[idx_sample[j], idx_backdoor[j]]})

    def save_information(self):
        return {
            'weights': self.state_dict(),
            'init_weights': [self.init_input_layer_weight, self.init_input_layer_bias],
            'possible_images': self.possible_images,
            'arch': self.arch,
            'preprocess_information': self.preprocess_information,
            'num_backdoors': self.num_backdoors
        }

    def load_information(self, infor_dict):
        self.load_state_dict(infor_dict['weights'])
        self.init_input_layer_weight, self.init_input_layer_bias = infor_dict['init_weights']
        self.possible_images = infor_dict['possible_images']
        self.arch = infor_dict['arch']
        self.preprocess_information = infor_dict['preprocess_information']
        self.num_backdoors = infor_dict['num_backdoors']

    def backdoor_initialize(self, num_backdoor, baits_info=None, intermediate_info=None, output_info=None):
        # baits_info: (bait, threshold, possible_classes)
        self.num_backdoors = num_backdoor
        # input layer
        bait, threshold, possible_classes = baits_info
        for j in range(self.num_backdoors):
            self.input_layer.weight.data[j, :] = bait[j]
            self.input_layer.bias.data[j] = -1.0 * threshold[j]

        # intermediate layer
        if intermediate_info is None:
            intermediate_multiplier = 1.0
            intermediate_noise_threshold = 0.0
        else:
            intermediate_multiplier = intermediate_info['multiplier']
            intermediate_noise_threshold = intermediate_info['noise_threshold']
        for j in range(self.num_backdoors):
            self.intermediate_layer.weight.data[:, j] = 0.0
            self.intermediate_layer.weight.data[j, :] = 0.0
            self.intermediate_layer.weight.data[j, j] = intermediate_multiplier
            self.intermediate_layer.bias.data[j] = -1.0 * intermediate_noise_threshold

        approach = output_info['approach']
        num_classes = self.output_layer.out_features
        classes = [j for j in range(num_classes)]
        if approach == 'random_connect':
            for j in range(self.num_backdoors):
                self.output_layer.weight.data[:, j] = 0.0
                target_class = random.choice(classes)
                self.output_layer.weight.data[target_class, j] = output_info['multiplier']

        elif approach == 'wrong_class':
            for j in range(self.num_backdoors):
                self.output_layer.weight.data[:, j] = 0.0
                possible_classes_this_bait = possible_classes[j]
                wrong_classes = cal_set_difference_seq(num_classes, possible_classes_this_bait)
                wrong_class = random.choice(wrong_classes)
                self.output_layer.weight.data[wrong_class, j] = output_info['multiplier']

        elif approach == 'random_gaussian':
            nn.init.xavier_normal_(self.output_layer.weight)

        else:
            pass

        self.init_input_layer_weight = self.input_layer.weight.detach().clone()
        self.init_input_layer_bias = self.input_layer.bias.detach().clone()
        self.possible_images = [[] for j in range(self.num_backdoors)]

    def reconstruct_images(self, c, h, w):
        weight0, bias0 = self.init_input_layer_weight.detach().clone(), self.init_input_layer_bias.detach().clone()
        weight1, bias1 = self.input_layer.weight.detach().clone(), self.input_layer.bias.detach().clone()
        delta_weight = weight1 - weight0
        delta_bias = bias1 - bias0
        reconstruct_lst = []
        for j in range(self.num_backdoors):
            image_flat = delta_weight[j] / (delta_bias[j] + 1e-8)
            image = image_flat.reshape(c, h, w)
            reconstruct_lst.append(image)
        return reconstruct_lst

    def show_possible_images(self, approach):
        out_images = []
        get_max = lambda x: x['act']
        for j in range(self.num_backdoors):
            possible_images_this_door = self.possible_images[j]
            print(f'backdoor:{j}, number:{len(possible_images_this_door)}')
            if approach == 'first':
                out_images.append(possible_images_this_door[0]['image'])
            elif approach == 'largest':
                image_max = max(possible_images_this_door, key=get_max)
                out_images.append(image_max['image'])
            elif approach == 'mix':
                if len(possible_images_this_door) > 1:
                    out_images.append(0.5 * torch.ones(self.arch['input_size']))
                else:
                    out_images.append(possible_images_this_door[0]['image'])
            else:
                pass
        return out_images

    def show_backdoor_change(self):
        weight0, bias0 = self.init_input_layer_weight.detach().clone().to('cpu'), self.init_input_layer_bias.detach().clone().to('cpu')
        weight1, bias1 = self.input_layer.weight.detach().clone().to('cpu'), self.input_layer.bias.detach().clone().to('cpu')
        delta_weight, delta_bias = weight1 - weight0, bias1 - bias0
        delta_bias_printable = ','.join(['{:.2e}'.format(delta_bias_this_door.item())
                                         for delta_bias_this_door in delta_bias[:self.num_backdoors]])
        return delta_bias_printable


def native_preprocess(images, preprocess_information=None):
    if preprocess_information is None:
        x = images.reshape(len(images), -1)
    else:
        x = None
    return x


def native_bait_selector(baits_candidate, dataloader4estimate, quantile=0.001,  select_info=None,
                         preprocess_information=None):

    num_input = baits_candidate.shape[1]
    # select
    if select_info is not None:
        largest_correlation = select_info.get('largest_correlation', None)
        if largest_correlation is not None:
            print('select by largest correlation')
            bait_satisfied = torch.zeros(0, num_input)

            for j in range(len(baits_candidate)):
                this_bait = baits_candidate[j].unsqueeze(dim=0)
                this_bait_nm = this_bait / this_bait.norm(keepdim=True, dim=1)
                correlation = (bait_satisfied / bait_satisfied.norm(keepdim=True, dim=1)) @ this_bait_nm.t()
                this_ok = torch.all(torch.lt(correlation, largest_correlation))
                if this_ok:
                    bait_satisfied = torch.cat([bait_satisfied, this_bait], dim=0)
        else:
            bait_satisfied = baits_candidate
    else:
        bait_satisfied = baits_candidate

    score_lst, label_lst = [], []
    for image, label in dataloader4estimate:
        x = native_preprocess(image, preprocess_information)
        score = x @ bait_satisfied.t()  # num_sample * num_bait
        score_lst.append(score)
        label_lst.append(label)
    all_score = torch.cat(score_lst)
    all_label = torch.cat(label_lst)

    q = torch.quantile(all_score, q=quantile, dim=0, keepdim=True)
    largest, _ = all_score.max(dim=0)
    is_active = (all_score > q)
    possible_classes = [all_label[is_active[:, j]] for j in range(is_active.shape[1])]

    return bait_satisfied, (q.squeeze(), largest), possible_classes


class EncoderMLP(nn.Module):
    def __init__(self, encoder=None, mlp_sizes=None, input_size=(3, 32, 32), num_classes=10, dropout=None,
                 return_intermediate=False):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.encoder = encoder
        self.return_intermediate = return_intermediate

        assert len(mlp_sizes) == 2, 'the mlp should have two layers'
        self.mlp_sizes = mlp_sizes
        if isinstance(dropout, float):
            self.mlp_1stpart = nn.Sequential(nn.Linear(self.get_num_features(), self.mlp_sizes[0]),
                                        nn.ReLU(), nn.Dropout(p=dropout))
            self.mlp_2ndpart = nn.Sequential(nn.Linear(self.mlp_sizes[0], self.mlp_sizes[1]),
                                        nn.ReLU(), nn.Dropout(p=dropout))
        else:
            self.mlp_1stpart = nn.Sequential(nn.Linear(self.get_num_features(), self.mlp_sizes[0]), nn.ReLU())
            self.mlp_2ndpart = nn.Sequential(nn.Linear(self.mlp_sizes[0], self.mlp_sizes[1]), nn.ReLU())
        self.probe = nn.Linear(self.mlp_sizes[1], self.num_classes)  # the structure is fixed after initialization
        # later change offer affect the value of each weight

        self.module_names = ['encoder', 'mlp_1stpart', 'mlp_2ndpart', 'probe']

    def forward(self, images):
        features = self.encoder(images)
        u = self.mlp_1stpart(features)
        v = self.mlp_2ndpart(u)
        z = self.probe(v)

        if self.return_intermediate:
            return z, u.clone().detach(), v.clone().detach()
        else:
            return z

    def get_num_features(self, inputs=None):
        if inputs is None:
            inputs = torch.randn(self.input_size)
            inputs = inputs.unsqueeze(dim=0)
        y = self.encoder(inputs)
        assert y.dim() == 2, 'the output of encoder does not meet the requirement of MLP'
        return y.shape[1]

    def load_weight(self, path_to_weights, which_module=None):
        weights = torch.load(path_to_weights, map_location='cpu')
        assert isinstance(weights, dict), 'the weight should be a dict'
        assert isinstance(which_module, str), "only accept string as modules' name"
        if which_module is None:
            selected_module = self.module_names
        elif isinstance(which_module, str) and which_module == 'mlp':
            selected_module = self.module_names[1:]
        elif isinstance(which_module, str) and which_module == 'other_than_probe':
            selected_module = self.module_names[:-1]
        else:
            selected_module = [which_module]

        for module in selected_module:
            getattr(self, module).load_state_dict(weights[module])

    def save_weight(self):
        state_dicts = {}
        for module in self.module_names:
            state_dicts[module] = getattr(self, module).state_dict()
        return state_dicts

    def module_parameters(self, module='mlp'):
        if module == 'encoder':
            params = [param for name, param in self.encoder.named_parameters()]
        elif module == 'mlp':
            params = [param for mlp_module in self.module_names[1:] for name, param in getattr(self, mlp_module).named_parameters()]
        elif module == 'other_than_probe':
            params = [param for name, param in self.named_parameters() if name not in ['probe.weight', 'probe.bias']]
        elif module == 'probe':
            params = [param for name, param in self.named_parameters() if name in ['probe.weight', 'probe.bias']]
        else:
            params = [param for name, param in getattr(self, module).named_parameters()]
        return params

    def activate_gradient_or_not(self, module='encoder', is_activate=False):
        params = self.module_parameters(module=module)
        for param in params:
            param.requires_grad = is_activate


def record_step_info(act_sample, act_bkd_idx, bkd_counter, values_bkd, eps=1e-5):
    act_sample = torch.tensor(act_sample)
    act_bkd_idx = torch.tensor(act_bkd_idx)
    if len(act_bkd_idx) > 0:
        bkd_keys = list(bkd_counter.keys())
        info_step = {'bkd_idx': bkd_keys, 'counter': [bkd_counter[key] for key in bkd_keys], 'bkd_value': []}
        for key in info_step['bkd_idx']:
            activate_this_key = torch.eq(act_bkd_idx, key)
            activate_times = torch.sum(activate_this_key)
            if activate_times == 0:
                assert False, 'WRONG, does not match'
            if activate_times == 1:
                value = values_bkd[act_sample[activate_this_key], key].squeeze().item()
                info_step['bkd_value'].append(value)
            else:
                values = values_bkd[act_sample[activate_this_key], act_bkd_idx[activate_this_key]]

                if values.std() < eps:
                    info_step['bkd_value'].append(values[0].item())
                else:
                    assert False, 'two different samples activate a backdoor'
    else:
        info_step = {}
    return info_step


class DiffPrvBackdoorRegistrar:
    def __init__(self, num_bkd=None, m_u=None, m_v=None, indices_bkd_u=None, indices_bkd_v=None, target_image_label=None):
        # TODO: many assert to guarantee correctness.
        # TODO: deal with different input dimension of backdoor
        # TODO: how to make sure that u,v, target are one-to-one correspondence
        # this should have the ability to generate target_image and target labels

        if num_bkd is not None and m_u is not None and m_v is not None:
            self.num_bkd = num_bkd
            self.m_u, self.m_v = m_u, m_v
            assert m_u > num_bkd and m_v > num_bkd,  'the number of features have to be larger than the number of backdoors'
        else:
            self.num_bkd = None
            self.m_u, self.m_v = None, None

        indices_bkd_u = [indices_bkd_u] if isinstance(indices_bkd_u, int) else indices_bkd_u
        indices_bkd_v = [indices_bkd_v] if isinstance(indices_bkd_v, int) else indices_bkd_v
        if indices_bkd_u is not None and indices_bkd_v is not None:
            self.indices_bkd_u, self.indices_bkd_v = torch.tensor(indices_bkd_u), torch.tensor(indices_bkd_v)
            assert isinstance(self.indices_bkd_u, torch.Tensor) and self.indices_bkd_u.dim() == 1, 'the dimension of indices should be 1'
            assert isinstance(self.indices_bkd_v, torch.Tensor) and self.indices_bkd_v.dim() == 1, 'the dimension of indices should be 1'
            assert self.num_bkd == len(self.indices_bkd_u) and self.num_bkd == len(self.indices_bkd_v), 'the number of backdoors is not correct'
            assert torch.logical_and(torch.all(self.indices_bkd_u < m_u), torch.all(self.indices_bkd_u >= 0)), 'the indices out of range'
            assert torch.logical_and(torch.all(self.indices_bkd_v < m_v), torch.all(self.indices_bkd_v >= 0)), 'the indices out of range'

            self.indices_others_u, self.indices_others_v = cal_set_difference_seq(m_u, self.indices_bkd_u), cal_set_difference_seq(m_v, self.indices_bkd_v)
        else:
            self.indices_bkd_u, self.indices_bkd_v = None, None
            self.indices_others_u, self.indices_others_v = None, None

        self.backdoor_arch_info = {'num_bkd': num_bkd, 'm_u': m_u, 'm_v': m_v, 'indices_bkd_u': indices_bkd_u,
                                   'indices_bkd_v': indices_bkd_v, 'target_image_label': target_image_label}

        self.u_act_log = []
        self.v_act_log = []
        self.inner_outputs_cache = []
        self.v_cache = []
        self.bu_bkd_log = []

        self.epoch = -1
        self.eps = 1e-5

        if target_image_label is not None:
            self.target_images = [target_img_lb[0] for target_img_lb in target_image_label]
            self.target_labels = [target_img_lb[1] for target_img_lb in target_image_label]
        else:
            self.target_images = None
            self.target_labels = None

    def save_information(self):
        return {
            'num_bkd': self.num_bkd, 'm_u': self.m_u, 'm_v': self.m_v,
            'indices_bkd_u': self.indices_bkd_u , 'indices_bkd_v': self.indices_bkd_v, 'indices_others_u': self.indices_others_u, 'indices_others_v': self.indices_others_v,
            'u_act_log':self.u_act_log, 'v_act_log': self.v_act_log, 'inner_outputs_cache':self.inner_outputs_cache ,'v_cache':self.v_cache, 'bu_bkd_log':self.bu_bkd_log,
            'epoch':self.epoch, 'eps': self.eps,
            'target_images':self.target_images, 'target_labels':self.target_labels,
            'backdoor_arch_info': self.backdoor_arch_info
        }

    def load_information(self, info_dict):
        for key in info_dict.keys():
            setattr(self, key, info_dict[key])

    def update_epoch(self, epoch):
        assert epoch - self.epoch == 1, 'wrong update'
        self.epoch = epoch
        self.u_act_log.append([])
        self.v_act_log.append([])
        self.bu_bkd_log.append([])

    def update_output_log(self, u, v):
        # u, v: num_samples * num_features
        assert u.shape[1] == self.m_u and v.shape[1] == self.m_v, 'the number of features does not match'
        u_bkd, v_bkd = u[:, self.indices_bkd_u], v[:, self.indices_bkd_v]
        assert u_bkd.dim() == 2 and v_bkd.dim() == 2, 'the u,v should have 2 dimension'

        act_u_sample, act_u_bkd = torch.nonzero(u_bkd > self.eps, as_tuple=True)  # a sample can only activate a door, a door can be activated by severa samples
        act_u_sample, act_u_bkd = act_u_sample.tolist(), act_u_bkd.tolist()
        act_v_sample, act_v_bkd = torch.nonzero(v_bkd > self.eps, as_tuple=True)
        act_v_sample, act_v_bkd = act_v_sample.tolist(), act_v_bkd.tolist()
        assert len(act_u_sample) == len(set(act_u_sample)), 'WRONG: there is some sample that activate more than one backdoor'
        assert len(act_v_sample) == len(set(act_v_sample)), 'WRONG: there is some sample that activate more than one backdoor'
        u_bkd_counter = Counter(act_u_bkd)
        v_bkd_counter = Counter(act_v_bkd)

        info_step_u = record_step_info(act_sample=act_u_sample, act_bkd_idx=act_u_bkd, bkd_counter=u_bkd_counter,
                                       values_bkd=u_bkd, eps=self.eps)
        # value = values_bkd[act_sample[activate_this_key], key].squeeze().item()
        self.u_act_log[self.epoch].append(info_step_u)

        info_step_v = record_step_info(act_sample=act_v_sample, act_bkd_idx=act_v_bkd, bkd_counter=v_bkd_counter,
                                       values_bkd=v_bkd, eps=self.eps)
        self.v_act_log[self.epoch].append(info_step_v)

    def collect_inner_state(self, inner_output):
        self.inner_outputs_cache.append(inner_output)

    def update_log_logical(self):
        u_lst, v_lst = [], []
        for inner_output in self.inner_outputs_cache:
            u, v = inner_output
            u_lst.append(u)
            v_lst.append(v)
        u = torch.cat(u_lst)
        v = torch.cat(v_lst)
        self.inner_outputs_cache = []
        self.update_output_log(u, v)

    def update_state(self, bu):
        bu = bu.detach().clone()
        self.bu_bkd_log[self.epoch].append(bu[self.indices_bkd_u])

    def _uvlog2array(self, log, is_stitch_overall=False, is_activation_counter=False):
        lst = []
        for log_this_epoch in log:
            lst_this_epoch = []
            for log_this_step in log_this_epoch:
                signal_this_step = torch.zeros(self.num_bkd)
                if len(log_this_step.keys()) > 0:
                    signal_this_step[log_this_step['bkd_idx']] = torch.tensor(log_this_step['counter']).float() if is_activation_counter else torch.tensor(log_this_step['bkd_value'])
                lst_this_epoch.append(signal_this_step)
            array_this_epoch = torch.stack(lst_this_epoch)
            lst.append(array_this_epoch)
        if is_stitch_overall:
            return torch.cat(lst)
        else:
            return lst

    def _bulog2array(self, log, is_stitch_overall=False):
        lst = []
        for log_this_epoch in log:
            lst_this_epoch = []
            for log_this_step in log_this_epoch:
                lst_this_epoch.append(log_this_step)
            array_this_epoch = torch.stack(lst_this_epoch)
            lst.append(array_this_epoch)
        if is_stitch_overall:
            return torch.cat(lst)
        else:
            return lst

    def get_change_by_activation(self, activation_count=None, ignore_last=False):
        if ignore_last:
            u_act_log = self.u_act_log[:-1]
            bu_bkd_log = self.bu_bkd_log[:-1]
        else:
            u_act_log = self.u_act_log
            bu_bkd_log = self.bu_bkd_log

        activation_array = self._uvlog2array(u_act_log, is_stitch_overall=True, is_activation_counter=True)
        bu_bkd_array = self._bulog2array(bu_bkd_log, is_stitch_overall=True)
        assert len(bu_bkd_array) - len(activation_array) == 1, f'should have n+1 states and n passes: {len(bu_bkd_array)},{len(activation_array)}'

        is_select = (activation_array == activation_count)
        delta_bu_bkd = bu_bkd_array[1:] - bu_bkd_array[:-1]
        delta_bu_select = delta_bu_bkd[is_select]
        return delta_bu_select

    def output_delta_b(self):
        bu_bkd_array = self._bulog2array(self.u_act_log, is_stitch_overall=True)
        return bu_bkd_array[-1] - bu_bkd_array[0]


class DiffPrvGradRegistrar:
    def __init__(self, backdoor_weight_name=None, backdoor_indices=None, backdoor_arch_info=None):
        self.backdoor_weight_name = backdoor_weight_name
        self.backdoor_indices = backdoor_indices
        self.grad_log = []
        self.v2class_log = []
        self.epoch = -1
        self.backdoor_arch_info = backdoor_arch_info  # num_bkd,indices_bkd_u,indices_bkd_v,m_u,m_v,target_image_label

    def save_information(self):
        return {
            'backdoor_weight_name': self.backdoor_weight_name,
            'backdoor_indices': self.backdoor_indices,
            'grad_log': self.grad_log,
            'v2class_log': self.v2class_log,
            'backdoor_arch_info': self.backdoor_arch_info
        }

    def load_information(self, info_dict):
        for key in info_dict.keys():
            setattr(self, key, info_dict[key])

    def update_grad_log(self, model, approach='summed_grad'):
        wgt = eval(f'model.{self.backdoor_weight_name}')
        grad = getattr(wgt, approach)
        grad_at_backdoor = eval(f'grad[{self.backdoor_indices}]')
        grad_at_backdoor = grad_at_backdoor.unsqueeze(dim=0) if grad_at_backdoor.dim() == 0 else grad_at_backdoor
        self.grad_log[self.epoch].append(grad_at_backdoor)

    def update_v2class_log(self, model):
        v2class_all = model.probe.weight.detach().clone()
        v2_class_bkd = v2class_all[:, self.backdoor_arch_info['indices_bkd_v']]
        self.v2class_log[self.epoch].append(v2_class_bkd)

    def update_epoch(self, epoch):
        assert epoch - self.epoch == 1, 'wrong update'
        self.epoch = epoch
        self.grad_log.append([])
        self.v2class_log.append([])

    def output_gradient_log(self, byepoch=False):
        output_grads = []
        for info_by_epoch in self.grad_log:
            grad_this_epoch = torch.stack(info_by_epoch)
            output_grads.append(grad_this_epoch)
        if byepoch:
            return output_grads
        else:
            return torch.cat(output_grads)

    def output_v2class_log(self):
        all_weights = torch.stack([item for logs in self.v2class_log for item in logs])
        return all_weights

    def check_v2class_largest(self):
        # assume there is only one backdoor
        all_weights = self.output_v2class_log()
        idx_max = all_weights[0].argmax()
        idx_max_during_training = all_weights.argmax(dim=1)
        print(f'total updates:{len(idx_max_during_training)}, max weight still largest{torch.sum(torch.eq(idx_max_during_training, idx_max))}') # idx_max is a scalar

    def get_largest_correct_classes(self):
        all_weights = self.output_v2class_log()
        idx_max = all_weights[0].argmax()
        labels = [label for image, label in self.backdoor_arch_info['target_image_label']]
        return idx_max, labels

    def count_nonzero_grad_by_epoch(self, noise_thres=1e-3):
        grad_by_epoch = self.output_gradient_log(byepoch=True)
        for i, grad_this_epoch in enumerate(grad_by_epoch):
            grad_this_epoch_abs = torch.abs(grad_this_epoch)
            is_non_noise = torch.ge(grad_this_epoch_abs, noise_thres)
            print(f'epoch:{i}, non-noise gradient{is_non_noise.sum(dim=0).tolist()}')


class InitEncoderMLP(EncoderMLP):
    def __init__(self, encoder=None, mlp_sizes=None, input_size=(3, 32, 32), num_classes=10):
        super().__init__(encoder=encoder, mlp_sizes=mlp_sizes, input_size=input_size, num_classes=num_classes, dropout=None)

    def forward(self, images):
        features = self.encoder(images)
        u = self.mlp_1stpart(features)
        v = self.mlp_2ndpart(u)
        z = self.probe(v)
        return z, (u.clone().detach(), v.clone().detach())

    def initialize_backdoor(self, encoder_scaling_module_idx=-1, baits=None, thresholds=None, passing_threshold=None,
                            multipliers=None, backdoor_registrar=None):
        u_indices_bkd,  v_indices_bkd = backdoor_registrar.backdoor_arch_info['indices_bkd_u'], backdoor_registrar.backdoor_arch_info['indices_bkd_v']

        idx_module, scaling_multiplier = encoder_scaling_module_idx, multipliers.get('encoder', 1.0),
        assert isinstance(self.encoder[idx_module], nn.Conv2d)
        self.encoder[idx_module].weight.data = scaling_multiplier * self.encoder[idx_module].weight.detach().clone()
        self.encoder[idx_module].bias.data = scaling_multiplier * self.encoder[idx_module].bias.detach().clone()

        threshold_multiplier = multipliers.get('encoder', 1.0) * multipliers.get('bait', 1.0)
        self._pass_ft_build_act(self.mlp_1stpart[0], indices_bkd=u_indices_bkd, baits=baits, thresholds=thresholds,
                                ft_passing_multiplier=multipliers.get('features_passing', 1.0), bait_multiplier=multipliers.get('bait', 1.0), threshold_multiplier=threshold_multiplier)

        nn.init.xavier_normal_(self.mlp_2ndpart[0].weight)
        self.mlp_2ndpart[0].bias.data[:] = 0.
        self._lock_ft_pass_act(self.mlp_2ndpart[0], u_indices_bkd=u_indices_bkd, v_indices_bkd=v_indices_bkd, passing_threshold=passing_threshold,
                               lock_multiplier=multipliers.get('features_lock', 1.0), act_passing_multiplier=multipliers.get('activation_passing', 1.0), threshold_multiplier=threshold_multiplier)

    def initialize_crafted_head(self, backdoor_registrar, act_connect_multiplier=1.0):
        num_bkd = backdoor_registrar.backdoor_arch_info['num_bkd']
        target_image_label = backdoor_registrar.backdoor_arch_info['target_image_label']
        target_labels = [label for img, label in target_image_label]
        v_indices_bkd = backdoor_registrar.backdoor_arch_info['indices_bkd_v']
        classes_connect = []
        for j in range(num_bkd):
            complement_set = cal_set_difference_seq(self.num_classes, target_labels[j])
            complement_set = complement_set.tolist()
            classes_connect.append(random.choice(complement_set))
        nn.init.xavier_normal_(self.probe.weight)
        self.probe.bias.data[:] = 0.
        self._act_connect(self.probe, indices_bkd=v_indices_bkd, wrong_classes=classes_connect,
                          act_connect_multiplier=act_connect_multiplier)

    def initialize_random_head(self, backdoor_registrar, gain=1.0, threshold=0.0, num_trial=100):
        target_image_label = backdoor_registrar.backdoor_arch_info['target_image_label']
        target_labels = torch.tensor([imglb[1] for imglb in target_image_label])
        v_indices_bkd = backdoor_registrar.backdoor_arch_info['indices_bkd_v']

        for j in range(num_trial):
            nn.init.xavier_normal_(self.probe.weight, gain=gain)
            weights = self.probe.weight.detach().clone()
            values, indices = weights.topk(2, dim=0)
            gap = values[0] - values[1]
            is_satisfy = torch.ge(gap, threshold)
            print(f'proportion,{torch.sum(is_satisfy) / len(gap)}')

            largest_class = indices[0, v_indices_bkd]
            second_largest_class = indices[1, v_indices_bkd]
            backdoor_satisfy_class = torch.all(torch.logical_not(torch.eq(largest_class, target_labels)))
            is_gap_bkd_satisfy = is_satisfy[v_indices_bkd]
            backdoor_satisfy_gap = torch.all(is_gap_bkd_satisfy)
            if backdoor_satisfy_class and backdoor_satisfy_gap:
                print(f'largest:{weights[largest_class, v_indices_bkd]}, second largest:{weights[second_largest_class, v_indices_bkd]}, class:{weights[target_labels, v_indices_bkd]}')
                return

        print('!!!NO SUITABLE WEIGHTS!!!')

    def _pass_ft_build_act(self, module, indices_bkd, baits, thresholds,
                           ft_passing_multiplier=1.0, bait_multiplier=1.0, threshold_multiplier=1.0):
        assert isinstance(module, nn.Linear), 'the module should be linear'
        assert len(indices_bkd) == len(baits) and len(baits) == len(thresholds), 'backdoor, bait, threshold should be one-to-one correspondence'

        module.weight.data[:] = 0.0
        module.bias.data[:] = 0.0

        indices_ft = cal_set_difference_seq(module.out_features, indices=indices_bkd)
        for idx in indices_ft:
            module.weight.data[idx, idx] = ft_passing_multiplier

        for j, idx in enumerate(indices_bkd):
            module.weight.data[idx, :] = bait_multiplier * baits[j]
            module.bias.data[idx] = - threshold_multiplier * thresholds[j]

    def _lock_ft_pass_act(self, module, u_indices_bkd, v_indices_bkd, passing_threshold,
                          lock_multiplier=1.0, act_passing_multiplier=1.0, threshold_multiplier=1.0):
        assert isinstance(module, nn.Linear), 'the module should be linear'
        assert len(v_indices_bkd) == len(u_indices_bkd) and len(u_indices_bkd) == len(passing_threshold), 'the number of input backdoors should be the same as the output and threshold'

        indices_ft_v = cal_set_difference_seq(module.out_features, indices=v_indices_bkd)
        for idx in indices_ft_v:
            module.weight.data[idx, u_indices_bkd] = - lock_multiplier

        module.weight.data[v_indices_bkd] = 0.0
        module.weight.data[v_indices_bkd, u_indices_bkd] = act_passing_multiplier
        module.bias.data[v_indices_bkd] = - passing_threshold * act_passing_multiplier * threshold_multiplier # passing threshold should be large enough to keep u* v* activation at the same time during training

    def _act_connect(self, module, indices_bkd, wrong_classes, act_connect_multiplier=1.0):
        assert isinstance(module, nn.Linear), 'the module should be linear'
        assert len(indices_bkd) == len(wrong_classes), 'for vanilla probe, the number of backdoor should be the same as the number of classes'
        for j, idx in enumerate(indices_bkd):
            module.weight.data[:, idx] = 0.0
            module.bias.data[:] = 0.0
            module.weight.data[wrong_classes[j], idx] = act_connect_multiplier




if __name__ == '__main__':
    pass





