import torch
import torch.nn as nn
from tools import cal_set_difference_seq
import random


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
            self.mlp_2ndpart = nn.Sequential(nn.Linear(self.mlp_size[0], self.mlp_sizes[1]),
                                        nn.ReLU(), nn.Dropout(p=dropout))
        else:
            self.mlp_1stpart = nn.Sequential(nn.Linear(self.get_num_features(), self.mlp_sizes[0]), nn.ReLU())
            self.mlp_2ndpart = nn.Sequential(nn.Linear(self.mlp_size[0], self.mlp_sizes[1]), nn.ReLU())
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


class DiffPrvBackdoorRegistrar:
    def __init__(self, indices_bkd_u=-1, indices_bkd_v=-1, m_u=None, m_v=None, targets=None, labels=None):
        # this should have the ability to generate target_image and target labels.
        self.indices_bkd_u, self.indices_bkd_v = indices_bkd_u, indices_bkd_v
        self.indices_others_u, self.indices_others_v = cal_set_difference_seq(m_u, self.indices_bkd_u), cal_set_difference_seq(m_v, self.indices_bkd_v)
        self.num_bkd = len(self.indices_bkd_u)
        # TODO: many assert to guarantee correctness.

        self.u_log = []
        self.v_log = []
        self.activation_log = []
        self.bu_bkd_log = []
        self.epoch = 0
        self.eps = 1e-5
        self.targets = targets
        self.labels = labels

    def update_output_log(self, u, v):
        # u_ft, v_ft = u[:, self.indices_others_u], v[:, self.indices_others_v]
        u_bkd, v_bkd = u[:, self.indices_bkd_u], v[:, self.indices_bkd_v]

        u_which_activated = (u_bkd > self.eps)
        v_which_activated = (v_bkd > self.eps)
        if u_which_activated.dim() == 2 and v_which_activated.dim() == 2:
            self.activation_log.append({'epoch': self.epoch, 'u_bkd_idx': u_which_activated[:, 1], 'u_bkd': u_bkd[u_which_activated[:, 0], u_which_activated[:, 1]],
                                        'v_bkd_idx': v_which_activated[:, 1], 'v_bkd': v_bkd[v_which_activated[:, 0], v_which_activated[:, 1]]})
        else:
            self.activation_log.append({})

    def update_state(self, epoch, bu):
        bu = bu.detach().clone()
        self.epoch = epoch
        self.bu_bkd_log.append({'epoch': epoch, 'bu_bkd': bu[self.indices_bkd_u]})

    def log_to_table(self, which_log):
        if which_log == 'activation':
            activation_log = getattr(self, which_log + '_log')
            activation_table = torch.zeros(len(activation_log), self.num_bkd)
            for j in range(len(activation_log)):
                if len(self.bu_activation_log[j].keys()) > 0:
                    activation_table[j, self.bu_activation_log[j]['u_bkd_idx']] = 1.0
            activation_table = (activation_table > 0.5)
            return activation_table
        elif which_log == 'bu_bkd':
            bu_bkd_log = getattr(self, which_log + '_log')
            bu_bkd_table = torch.zeros(len(bu_bkd_log), self.num_bkd)
            for j in range(bu_bkd_log):
                bu_bkd_table[j] = bu_bkd_log[j]['bu_bkd']
            return bu_bkd_table

    def show_activation_information(self):
        epoch = 0
        lst_epochs = [[]]
        for j in range(len(self.activation_log)):
            activation_this_step = self.activation_log[j]
            if len(activation_this_step.keys()) > 0:
                if activation_this_step['epoch'] != epoch:
                    epoch = activation_this_step['epoch']
                    lst_epochs.append([])
                lst_epochs[epoch].append(activation_this_step)
        lst_u_byepoch = []
        lst_v_byepoch = []
        for info_epoch in lst_epochs:
            activation_this_epoch_u = torch.zeros(len(info_epoch), self.num_bkd)
            activation_this_epoch_v = torch.zeros(len(info_epoch), self.num_bkd)
            for j in range(len(info_epoch)):
                if len(info_epoch[j].keys()) > 0:
                    activation_this_epoch_u[j, info_epoch[j]['u_bkd_idx']] = info_epoch[j]['u_bkd']
                    activation_this_epoch_v[j, info_epoch[j]['v_bkd_idx']] = info_epoch[j]['v_bkd']
            lst_u_byepoch.append(activation_this_epoch_u)
            lst_v_byepoch.append(activation_this_epoch_v)

        return lst_u_byepoch, lst_v_byepoch

    def get_change_by_activation(self):
        print(f'the length of activation log is {len(self.activation_log)}')
        print(f'the length of bu bkd log is {len(self.bu_bkd_log)}')  # bu_log should be activation log + 1
        activation_table = self.log_to_table('activation')
        bu_bkd_table = self.log_to_table('bu_bkd')
        delta_bu_bkd = bu_bkd_table[1:] - bu_bkd_table[:-1]
        activation_change = delta_bu_bkd[activation_table]
        inactivation_change = delta_bu_bkd[torch.logical_not(activation_table)]
        print(f'activation change mean value:{activation_change.mean()}, variance:{activation_change.var()}')
        print(f'activation change mean value:{inactivation_change.mean()}, variance:{inactivation_change.var()}')
        return activation_change, inactivation_change


class DiffPrvBackdoorMLP(EncoderMLP):
    def __init__(self, encoder=None, mlp_sizes=None, input_size=(3, 32, 32), num_classes=10, backdoor_registrar=None):
        super(self, DiffPrvBackdoorMLP).__init__(encoder=encoder, mlp_sizes=mlp_sizes, input_size=input_size,
                                                 num_classes=num_classes, dropout=None)

        self.backdoor_registrar = backdoor_registrar

    def forward(self, images):
        features = self(images)
        u = self.mlp_1stpart(features)
        v = self.mlp_2ndpart(u)
        z = self.probe(v)
        self.backdoor_registrar.update_output_log(u, v)
        return z

    def update_state(self, epoch):
        self.backdoor_registrar.update_state(epoch, self.mlp_1stpart[0].bias)

    def vanilla_initialize(self, encoder_scaling_module_idx=-1, weights=None, thresholds=None, passing_threshold=None, factors=None):
        # TODO: requires no gradient
        # requires no grad for encoder
        self.scale_encoder_output(idx_module=encoder_scaling_module_idx, scaling_factor=factors.get('encoder', 1.0))
        self.pass_feature_and_build_activation(self.mlp_1stpart[0], indices_bkd=self.backdoor_registrar.indices_bkd_u,
                                               weights=weights, thresholds=thresholds, passing_scaling_factor=factors.get('features_passing', 1.0),
                                               weight_factor=factors.get('bait', 1.0))
        self.lock_and_pass_activation(indices_bkd_input=self.backdoor_registrar.indices_bkd_u, indices_bkd_output=self.backdoor_registrar.indices_bkd_v,
                                      lock_factor=factors.get('lock', 1.0), activation_passing_factor=factors.get('activation_passing', 1.0),
                                      passing_threshold=passing_threshold)

        classes_connect = []
        for j in range(self.backdoor_registrar.num_bkd):
            complement_set = cal_set_difference_seq(self.num_classes, self.backdoor_registrar.labels[j])
            complement_set = complement_set.tolist()
            classes_connect.append(random.choice(complement_set))

        self.edit_vanilla_probe(indices_bkd=self.backdoor_registrar.indices_bkd_v, classes=classes_connect,
                                activation_class_factor=factors.get('connect', 1.0))

        for param in self.encoder.parameters():
            param.requires_grad = False

    def scale_encoder_output(self, idx_module=-1, scaling_factor=1.0):
        self.encoder[idx_module].weight.data = scaling_factor * self.encoder[idx_module].weight.detach().clone()
        self.encoder[idx_module].bias.data = scaling_factor * self.encoder[idx_module].bias.detach().clone()

    def pass_feature_and_build_activation(self, module, indices_bkd, weights, thresholds, passing_scaling_factor=1.0, weight_factor=1.0):
        assert len(indices_bkd) == len(weights) and len(indices_bkd) == len(thresholds),  'the number of should bkd should be the same as baits & bias'

        for j in range(len(self.mlp_1stpart.bias)):
            module.weight.data[j] = 0.0
            if j < module.weight.shape[1]:
                module.weight.data[j, j] = passing_scaling_factor
            module.bias[j] = 0.0

        for j in range(len(indices_bkd)):
            self.mlp_1stpart[0].weight.data[indices_bkd[j], :] = weight_factor * weights[j]
            self.mlp_1stpart[0].bias.data[indices_bkd[j], :] = - weight_factor * thresholds[j]

    def lock_and_pass_activation(self, indices_bkd_input, indices_bkd_output, lock_factor=1.0, activation_passing_factor=1.0, passing_threshold=0.0):
        # edit features
        assert len(indices_bkd_output) == len(indices_bkd_output), 'the number of input backdoors should be the same as the output backdoors'
        self.mlp_2ndpart[0].weight.data[:, indices_bkd_input] = - lock_factor

        self.mlp_2ndpart[0].weight.data[indices_bkd_output] = 0.0
        self.mlp_2ndpart[0].weight.data[indices_bkd_output, indices_bkd_input] = activation_passing_factor
        self.mlp_2ndpart[0].bias.data[indices_bkd_output] = - passing_threshold * activation_passing_factor  # passing threshold should be large enough to keep u* v* activation at the same time during training

    def edit_vanilla_probe(self, indices_bkd, classes, activation_class_factor=1.0):
        assert len(indices_bkd) == len(classes), 'for vanilla probe, the number of backdoor should be the same as the number of classes'
        for j in range(len(indices_bkd)):
            idx_bkd = indices_bkd[j]
            self.probe.weight.data[:, idx_bkd] = 0.0
            self.probe.weight.data[classes[j], idx_bkd] = activation_class_factor






