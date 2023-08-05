import torch
import torch.nn as nn
from tools import cal_set_difference_seq
import random
from collections import Counter


class EncoderMLP(nn.Module):
    def __init__(self, encoder=None, mlp_sizes=None, input_size=(3, 32, 32), num_classes=10, dropout=None):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.encoder = encoder

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
        if which_module is None:
            which_module = self.module_names
        elif isinstance(which_module, str) and which_module == 'mlp':
            which_module = self.module_names[1:]
        elif isinstance(which_module, str):
            which_module = [which_module]
        else:
            assert isinstance(which_module, list), 'input module that cannot be understood'

        for module in which_module:
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
        else:
            params = [param for name, param in getattr(self, module).named_parameters()]
        return params

    def activate_gradient_or_not(self, module='encoder', is_activate=False):
        params = self.module_parameters(module=module)
        for param in params:
            param.requires_grad = is_activate


def record_step_info(act_sample, act_bkd_idx, bkd_counter, values_bkd, eps=1e-5):
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
    def __init__(self, num_bkd, m_u=None, m_v=None, indices_bkd_u=None, indices_bkd_v=None, target_image_label=None):
        # TODO: many assert to guarantee correctness.
        # TODO: deal with different input dimension of backdoor
        # TODO: how to make sure that u,v, target are one-to-one correspondence
        # TODO: what to do with the coupling of
        # this should have the ability to generate target_image and target labels.
        self.num_bkd = num_bkd
        self.m_u, self.m_v = m_u, m_v
        assert m_u > num_bkd and m_v > num_bkd,  'the number of features have to be larger than the number of backdoors'

        indices_bkd_u = [indices_bkd_u] if isinstance(indices_bkd_u, int) else indices_bkd_u
        indices_bkd_v = [indices_bkd_v] if isinstance(indices_bkd_v, int) else indices_bkd_v
        self.indices_bkd_u, self.indices_bkd_v = torch.tensor(indices_bkd_u), torch.tensor(indices_bkd_v)
        assert isinstance(self.indices_bkd_u, torch.Tensor) and self.indices_bkd_u.dim() == 1, 'the dimension of indices should be 1'
        assert isinstance(self.indices_bkd_v, torch.Tensor) and self.indices_bkd_v.dim() == 1, 'the dimension of indices should be 1'
        assert self.num_bkd == len(self.indices_bkd_u) and self.num_bkd == len(self.indices_bkd_v), 'the number of backdoors is not correct'
        assert torch.logical_and(torch.all(self.indices_bkd_u < m_u), torch.all(self.indices_bkd_u >= 0)), 'the indices out of range'
        assert torch.logical_and(torch.all(self.indices_bkd_v < m_v), torch.all(self.indices_bkd_v >= 0)), 'the indices out of range'

        self.indices_others_u, self.indices_others_v = cal_set_difference_seq(m_u, self.indices_bkd_u), cal_set_difference_seq(m_v, self.indices_bkd_v)

        self.u_act_log = []
        self.v_act_log = []
        self.bu_bkd_log = []

        self.epoch = -1
        self.eps = 1e-5
        self.target_images, self.target_labels = target_image_label

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
        assert u_bkd.dim() == 2 and v_bkd.dim() == 2, ''

        act_u_sample, act_u_bkd = torch.nonzero(u_bkd > self.eps, as_tuple=True) # a sample can only activate a door, a door can be activated by severa samples
        act_u_sample, act_u_bkd = act_u_sample.tolist(), act_u_bkd.tolist()
        act_v_sample, act_v_bkd = torch.nonzero(v_bkd > self.eps, as_tuple=True)
        act_v_sample, act_v_bkd = act_v_sample.tolist(), act_v_bkd.tolist()
        assert len(act_u_sample) == len(set(act_u_sample)), 'WRONG: there is some sample that activate more than one backdoor'
        assert len(act_v_sample) == len(set(act_v_sample)), 'WRONG: there is some sample that activate more than one backdoor'
        u_bkd_counter = Counter(act_u_bkd)
        v_bkd_counter = Counter(act_v_bkd)

        info_step_u = record_step_info(act_sample=act_u_sample, act_bkd_idx=act_u_bkd, bkd_counter=u_bkd_counter,
                                       values_bkd=u_bkd, eps=self.eps)
        self.u_act_log[self.epoch].append(info_step_u)

        info_step_v = record_step_info(act_sample=act_v_sample, act_bkd_idx=act_v_bkd, bkd_counter=v_bkd_counter,
                                       values_bkd=v_bkd, eps=self.eps)
        self.v_act_log[self.epoch].append(info_step_v)

    def update_state(self, bu):
        bu = bu.detach().clone()
        self.bu_bkd_log[self.epoch].append(bu[self.indices_bkd_u])

    def _uvlog2array(self, log, is_stitch_overall=False, is_activation_counter=False):
        lst = []
        for log_this_epoch in log:
            lst_this_epoch = []
            for log_this_step in log_this_epoch:
                signal_this_step = torch.zeros(self.num_bkd)
                signal_this_step[log_this_step['bkd_idx']] = log_this_step['counter'] if is_activation_counter else log_this_step['bkd_value']
                lst_this_epoch.append(signal_this_step)
            array_this_epoch = torch.stack(lst_this_epoch)
            lst.append(array_this_epoch)
        if is_stitch_overall:
            return torch.stack(lst)
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
            return torch.stack(lst)
        else:
            return lst

    def get_change_by_activation(self, activation_count=None):
        activation_array = self._uvlog2array(self.u_act_log, is_stitch_overall=True, is_activation_counter=True)
        bu_bkd_array = self._bulog2array(self.bu_bkd_log, is_stitch_overall=True)
        assert len(bu_bkd_array) - len(activation_array) == 1, 'should have n+1 states and n passes'

        is_select = (activation_array == activation_count)
        delta_bu_bkd = bu_bkd_array[1:] - bu_bkd_array[:-1]
        delta_bu_select = delta_bu_bkd[is_select]
        return delta_bu_select

    def output_delta_b(self):
        bu_bkd_array = self._bulog2array(self.u_act_log, is_stitch_overall=True)
        return bu_bkd_array[-1] - bu_bkd_array[0]


class DiffPrvBackdoorMLP(EncoderMLP):
    def __init__(self, encoder=None, mlp_sizes=None, input_size=(3, 32, 32), num_classes=10, backdoor_registrar=None):
        super().__init__(encoder=encoder, mlp_sizes=mlp_sizes, input_size=input_size, num_classes=num_classes, dropout=None)
        self.backdoor_registrar = backdoor_registrar

    def forward(self, images):
        features = self.encoder(images)
        u = self.mlp_1stpart(features)
        v = self.mlp_2ndpart(u)
        z = self.probe(v)
        self.backdoor_registrar.update_output_log(u, v)

        return z

    def update_backdoor_registrar(self, backdoor_registrar):
        self.backdoor_registrar = backdoor_registrar

    def update_state(self):
        self.backdoor_registrar.update_state(self.mlp_1stpart[0].bias)

    def vanilla_initialize(self, encoder_scaling_module_idx=-1, baits=None, thresholds=None, passing_threshold=None, multipliers=None):
        u_indices_bkd,  v_indices_bkd = self.backdoor_registrar.indices_bkd_u, self.backdoor_registrar.indices_bkd_v

        idx_module, scaling_multiplier = encoder_scaling_module_idx, multipliers.get('encoder', 1.0),
        assert isinstance(self.encoder[idx_module], nn.Conv2d)
        self.encoder[idx_module].weight.data = scaling_multiplier * self.encoder[idx_module].weight.detach().clone()
        self.encoder[idx_module].bias.data = scaling_multiplier * self.encoder[idx_module].bias.detach().clone()

        self._pass_ft_build_act(self.mlp_1stpart[0], indices_bkd=u_indices_bkd, baits=baits, thresholds=thresholds,
                                ft_passing_multiplier=multipliers.get('features_passing', 1.0), bait_multiplier=multipliers.get('bait', 1.0))

        nn.init.xavier_normal_(self.mlp_2ndpart[0].weight)
        self.mlp_2ndpart[0].bias.data[:] = 0.
        self._lock_ft_pass_act(self.mlp_2ndpart[0], u_indices_bkd=u_indices_bkd, v_indices_bkd=v_indices_bkd, passing_threshold=passing_threshold,
                               lock_multiplier=multipliers.get('features_lock', 1.0), act_passing_multiplier=multipliers.get('activation_passing', 1.0))

        classes_connect = []
        for j in range(self.backdoor_registrar.num_bkd):
            complement_set = cal_set_difference_seq(self.num_classes, self.backdoor_registrar.labels[j])
            complement_set = complement_set.tolist()
            classes_connect.append(random.choice(complement_set))
        nn.init.xavier_normal_(self.probe.weight)
        self.probe.bias.data[:] = 0.
        self._act_connect(self.probe, indices_bkd=v_indices_bkd, wrong_classes=classes_connect, act_connect_multiplier=multipliers.get('act_connect', 1.0))

        self.activate_gradient_or_not('encoder', is_activate=False)
        self.activate_gradient_or_not('mlp', is_activate=True)

    def _pass_ft_build_act(self, module, indices_bkd, baits, thresholds,
                           ft_passing_multiplier=1.0, bait_multiplier=1.0):
        assert isinstance(module, nn.Linear), 'the module should be linear'
        assert len(indices_bkd) == len(baits) and len(baits) == len(thresholds), 'backdoor, bait, threshold should be one-to-one correspondence'

        module.weight.data[:] = 0.0
        module.bias.data[:] = 0.0

        indices_ft = cal_set_difference_seq(module.out_features, indices=indices_bkd)
        for idx in indices_ft:
            module.weight.data[idx, idx] = ft_passing_multiplier

        for j, idx in enumerate(indices_bkd):
            module.weight.data[idx, :] = bait_multiplier * baits[j]
            module.bias.data[idx] = - bait_multiplier * thresholds[j]

    def _lock_ft_pass_act(self, module, u_indices_bkd, v_indices_bkd, passing_threshold,
                          lock_multiplier=1.0, act_passing_multiplier=1.0):
        assert isinstance(module, nn.Linear), 'the module should be linear'
        assert len(v_indices_bkd) == len(u_indices_bkd) and len(u_indices_bkd) == len(passing_threshold), 'the number of input backdoors should be the same as the output and threshold'

        indices_ft_v = cal_set_difference_seq(module.out_features, indices=v_indices_bkd)
        for idx in indices_ft_v:
            module.weight.data[idx, u_indices_bkd] = lock_multiplier

        module.weight.data[v_indices_bkd] = 0.0
        module.weight.data[v_indices_bkd, u_indices_bkd] = act_passing_multiplier
        module.bias.data[v_indices_bkd] = - passing_threshold * act_passing_multiplier  # passing threshold should be large enough to keep u* v* activation at the same time during training

    def _act_connect(self, module, indices_bkd, wrong_classes, act_connect_multiplier=1.0):
        assert isinstance(module, nn.Linear), 'the module should be linear'
        assert len(indices_bkd) == len(wrong_classes), 'for vanilla probe, the number of backdoor should be the same as the number of classes'
        for j, idx in enumerate(indices_bkd):
            module.weight.data[:, idx] = 0.0
            module.bias.data[idx] = 0.0
            module.weight.data[wrong_classes[j], idx] = act_connect_multiplier
