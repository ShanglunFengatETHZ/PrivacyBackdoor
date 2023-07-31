import torch
import torch.nn as nn
from tools import cal_set_difference_seq
import random


class EncoderMLP(nn.Module):
    def __init__(self, encoder=None, mlp_sizes=None, input_size=(3, 32, 32), num_classes=10, dropout=None):
        super().__init__()
        self.input_size = input_size
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
        self.num_classes = num_classes
        self.probe = nn.Linear(self.mlp_sizes[1], self.num_classes)
        self.modules = ['encoder', 'mlp_1stpart', 'mlp_2ndpart', 'probe']

    def forward(self, images):
        features = self(images)
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
        if which_module is None:
            which_module = self.modules
        else:
            which_module = [which_module] if isinstance(which_module, str) and which_module in self.modules else which_module
            assert isinstance(which_module, list), 'the input of which module should be None or a list'
        for module in which_module:
            getattr(self, module).load_state_dict(weights)


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
        # requires no grad for encoder
        self.scale_encoder_output(idx_module=encoder_scaling_module_idx, scaling_factor=factors.get('encoder',1.0))
        self.pass_feature_and_build_activation(self, self.mlp_1stpart[0], indices_bkd=self.backdoor_registrar.indices_bkd_u,
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






