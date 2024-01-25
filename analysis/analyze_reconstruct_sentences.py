import argparse
import torch
import sys
import os
from transformers import BertTokenizer
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0])+'/src')
import src.tools as tools
from src.edit_bert import BertMonitor
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='INPUT the parameters of recovery')
    parser.add_argument('--path', type=str)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--verbose', type=bool, default=False, help='whether to print more information about activation times')

    return parser.parse_args()


def print_reconstruction_to_table(path, max_len,  posi_lst, posi_similarity, word_code_lst, smlar_1st_lst, alter_code_lst,
                                  smlar_2nd_lst, monitor, tokenizer, na_notation='', output_items=None):
    # used when output reconstructed sentences in a table
    with open(path, 'wt') as f:
        for j in range(len(word_code_lst)):
            word_code, smlar_1st, alter_code, smlar_2nd = word_code_lst[j], smlar_1st_lst[j], alter_code_lst[j], smlar_2nd_lst[j]
            posi_number, posi_similar = posi_lst[j], posi_similarity[j]

            word = []
            s1 = []
            alter = []
            s2 = []
            posi = []
            sm_posi = []

            for k in range(max_len):
                this_word = word_code[k]
                this_alter = alter_code[k]
                this_posi = posi_number[k]

                if this_posi > 0:
                    posi.append(str(int(this_posi)))
                    sm_posi.append(str(round(posi_similar[k].item(), 2)))
                else:
                    posi.append(na_notation)
                    sm_posi.append(na_notation)

                if this_word > 1000:  # ignore special character
                    new_word = monitor.get_text(tokenizer, [this_word], skip_special_tokens=False)
                    if new_word == ',':
                        new_word = '","'
                    word.append(new_word)
                    s1.append(str(round(smlar_1st[k], 2)))
                else:
                    word.append(na_notation)
                    s1.append(na_notation)

                if this_alter > 1000:  # ignore special character
                    new_alter = monitor.get_text(tokenizer, [this_alter], skip_special_tokens=False)
                    if new_alter == ',':
                        new_alter = '","'
                    alter.append(new_alter)
                    s2.append(str(round(smlar_2nd[k], 2)))
                else:
                    alter.append(na_notation)
                    s2.append(na_notation)

            word = ','.join(word)
            s1 = ','.join(s1)
            alter = ','.join(alter)
            s2 = ','.join(s2)
            posi = ','.join(posi)
            sm_posi = ','.join(sm_posi)

            if 'word' in output_items:
                print(f'word,{word}', file=f)
                print(f'similarity,{s1}', file=f)

            if 'position' in output_items:
                print(f'position,{posi}', file=f)
                print(f'similarity,{sm_posi}', file=f)

            if 'alternative' in output_items:
                print(f'alternative,{alter}', file=f)
                print(f'similarity,{s2}', file=f)


def print_readable_word(path, word_code_lst, monitor, tokenizer):
    with open(path, 'wt') as f:
        for word_code in word_code_lst:
            print(monitor.get_text(tokenizer, word_code, skip_special_tokens=True), file=f)


def print_final_presentation(path, reconstruction_lst, groundtruth_lst):
    path_reconstruction = f'{path}_reconstruction.txt'
    path_groundtruth = f'{path}_groundtruth.txt'
    with open(path_reconstruction, 'wt') as f:
        for k in range(len(reconstruction_lst)):
            print(f'{k}.  {reconstruction_lst[k]}', file=f)

    with open(path_groundtruth, 'wt') as g:
        for k in range(len(groundtruth_lst)):
            if len(groundtruth_lst[k]) == 0:
                print(f'{k}.  NONE', file=g)
            else:
                for i in range(len(groundtruth_lst[k])):
                    if i == 0:
                        print(f'{k}.  {groundtruth_lst[k][i]}', file=g)
                    else:
                        print(f'    {groundtruth_lst[k][i]}', file=g)


if __name__ == '__main__':

    args = parse_args()

    path = args.path
    save_path_pre = args.save_path
    verbose = args.verbose

    skip = True
    print_second = True
    monitor_information = torch.load(path, map_location='cpu')
    indices_ft = tools.indices_period_generator(num_features=768, head=64, start=0, end=8)
    monitor = BertMonitor()
    monitor.load_bert_monitor_information(monitor_information)

    position_code_all, _ = monitor._extract_information(block='embedding', submodule='position_embeddings')
    position_code_used = position_code_all[:, monitor.clean_position_indices]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    posi_lst = []
    posi_similarity = []
    word_code_lst = []
    smlar_1st_lst = []
    alter_code_lst = []
    smlar_2nd_lst = []

    delta_bkd_bias_printable, delta_bkd_bias, delta_bkd_estimate_printable, delta_bkd_estimate = monitor.get_backdoor_change()

    possible_sequences = monitor.show_possible_sequences(approach='semantics', logit_thres=1.0, verbose=verbose)

    reconstruction_lst = []
    groundtruth_lst = []

    for j in range(len(monitor.backdoor_indices)):
        bkd_indices = monitor.backdoor_indices[j]
        posi_code_this_seq, features_this_seq = monitor.get_update_a_sequence(indices_bkd_this_sequence=bkd_indices,
                                                                              target_entries=[monitor.clean_position_indices, indices_ft])

        posi_idx, similarity, second_similarity = monitor.get_digital_code(sequence=posi_code_this_seq, dictionary=position_code_used)
        word_code, similarity_1st, alternative_code, similarity_2nd = monitor.get_text_digital_code_this_sequence(features_this_seq, posi_idx, indices_ft,  centralize=True, output_zero=True)

        similarity = [round(x, 3) for x in similarity.tolist()]
        second_similarity = [round(x, 3) for x in second_similarity.tolist()]
        similarity_1st = [round(x, 3) for x in similarity_1st]
        similarity_2nd = [round(x, 3) for x in similarity_2nd]

        posi_lst.append(posi_idx)
        posi_similarity.append(similarity)
        word_code_lst.append(word_code)
        smlar_1st_lst.append(similarity_1st)
        alter_code_lst.append(alternative_code)
        smlar_2nd_lst.append(similarity_2nd)

        print(f'sequence{j} : position indices:{posi_idx}')
        print(f'position similarity:{similarity}')
        print(f'position second similarity:{second_similarity}')
        print(f'word indices:{word_code}')
        reconstruction = monitor.get_text(tokenizer, word_code, skip_special_tokens=skip)
        reconstruction_lst.append(reconstruction)
        print(f'reconstructed sentences:{reconstruction}')
        groundtruth_lst.append([])
        if len(possible_sequences[j]) > 10:
            print(f'ground truth: MESS')
            groundtruth_lst[j].append('MESS')
        elif len(possible_sequences[j]) == 0:
            print(f'ground truth: NONE')
            groundtruth_lst[j].append('NONE')
        else:
            for seq in possible_sequences[j]:
                if seq is not None:
                    possible_groundtruth = monitor.get_text(tokenizer, seq, skip_special_tokens=skip)
                    print(f'ground truth:{possible_groundtruth}')
                    groundtruth_lst[j].append(possible_groundtruth)
        print(f'word similarity:{similarity_1st}')

        print(f'alternative ground truth:{monitor.get_text(tokenizer, alternative_code, skip_special_tokens=skip)}')
        if print_second:
            print(f'alternative word similarity:{similarity_2nd}')
        print('\n')

    if save_path_pre is not None:
        print_final_presentation(save_path_pre, reconstruction_lst=reconstruction_lst, groundtruth_lst=groundtruth_lst)





