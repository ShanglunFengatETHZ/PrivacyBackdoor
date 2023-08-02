import torch


if __name__ == '__main__':
    path_to_registrar = None
    registrar = torch.load(path_to_registrar)
    ctivation_change, inactivation_change = registrar.get_change_by_activation()
    lst_u_byepoch, lst_v_byepoch = registrar.show_activation_information()