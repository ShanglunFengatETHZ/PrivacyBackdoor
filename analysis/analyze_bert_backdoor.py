import torch
import src.tools as tools
from src.edit_bert import BertMonitor
from transformers import BertTokenizer
import numpy as np


def print_reconstruction_to_table(path, max_len, word_code_lst, smlar_1st_lst, alter_code_lst, smlar_2nd_lst, monitor, tokenizer):
    with open(path, 'wt') as f:
        for j in range(len(word_code_lst)):
            word_code, smlar_1st, alter_code, smlar_2nd = word_code_lst[j], smlar_1st_lst[j], alter_code_lst[j], smlar_2nd_lst[j]
            word = []
            s1 = []
            alter = []
            s2 = []
            for k in range(max_len):
                this_word = word_code[k]
                this_alter = alter_code[k]
                if this_word > 1000:  # ignore special character
                    new_word = monitor.get_text(tokenizer, [this_word], skip_special_tokens=False)
                    if new_word == ',':
                        new_word = '","'
                    word.append(new_word)
                    s1.append(str(round(smlar_1st[k], 2)))
                else:
                    word.append('NA')
                    s1.append('NA')

                if this_alter > 1000:  # ignore special character
                    new_alter = monitor.get_text(tokenizer, [this_alter], skip_special_tokens=False)
                    if new_alter == ',':
                        new_alter = '","'
                    alter.append(new_alter)
                    s2.append(str(round(smlar_2nd[k], 2)))
                else:
                    alter.append('NA')
                    s2.append('NA')

            word = ','.join(word)
            s1 = ','.join(s1)
            alter = ','.join(alter)
            s2 = ','.join(s2)
            print(f'word,{word}', file=f)
            print(f'similarity,{s1}', file=f)
            print(f'alternative,{alter}', file=f)
            print(f'similarity,{s2}', file=f)


def print_readable_word(path, word_code, monitor, tokenizer):
    pass


if __name__ == '__main__':
    output_zero = True
    path = './weights/weak_shrink_monitor.pth'
    # save_path = './text_reconstruction.csv'
    skip = True
    print_second = True
    monitor_information = torch.load(path, map_location='cpu')
    indices_ft = tools.indices_period_generator(num_features=768, head=64, start=0, end=8)
    monitor = BertMonitor()
    monitor.load_bert_monitor_information(monitor_information)

    position_code_all, _ = monitor._extract_information(block='embedding', submodule='position_embeddings')
    position_code_used = position_code_all[:, monitor.clean_position_indices]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    word_code_lst = []
    smlar_1st_lst = []
    alter_code_lst = []
    smlar_2nd_lst = []

    for j in range(len(monitor.backdoor_indices)):
        bkd_indices = monitor.backdoor_indices[j]
        posi_code_this_seq, features_this_seq = monitor.get_update_a_sequence(indices_bkd_this_sequence=bkd_indices,
                                                                              target_entries=[monitor.clean_position_indices, indices_ft])

        posi_idx, similarity, second_similarity = monitor.get_digital_code(sequence=posi_code_this_seq, dictionary=position_code_used)
        word_code, similarity_1st, alternative_code, similarity_2nd = monitor.get_text_digital_code_this_sequence(features_this_seq, posi_idx,
                                                                                                indices_ft,  centralize=True, output_zero=output_zero)
        word_code_lst.append(word_code)
        smlar_1st_lst.append(similarity_1st)
        alter_code_lst.append(alternative_code)
        smlar_2nd_lst.append(similarity_2nd)

        print(f'sequence{j} : position indices:{posi_idx}')
        print(f'position similarity:{similarity}')
        print(f'position second similarity:{second_similarity}')
        print(f'word indices:{word_code}')
        print(f'text:{monitor.get_text(tokenizer, word_code, skip_special_tokens=skip)}')
        print(f'word similarity:{similarity_1st}')
        print(f'alternative text:{monitor.get_text(tokenizer, alternative_code, skip_special_tokens=skip)}')
        if print_second:
            print(f'word second similarity:{similarity_2nd}')
        print('\n')

    # print_reconstruction_to_table(save_path, 32, word_code_lst, smlar_1st_lst, alter_code_lst, smlar_2nd_lst, monitor, tokenizer)



