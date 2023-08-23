import torch
from tools import indices_period_generator
from transformers import BertTokenizer

if __name__ == '__main__':
    path = './weights/bert_v1_monitor.pth'
    indices_ft = indices_period_generator(num_features=768, head=64, start=0, end=8)
    monitor = torch.load(path)
    # feature_code, _ = None

    position_code_all, _ = monitor._extract_information(block='embedding', submodule='position_embeddings')
    position_code_used = position_code_all[:, monitor.clean_position_indices]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for bkd_indices in monitor.backdoor_indices:
        posi_code_this_seq, features_this_seq = monitor.get_update_a_sequence(indices_bkd_this_sequence=bkd_indices,
                                                                              target_entries=[monitor.clean_position_indices, indices_ft])

        posi_idx, similarity, second_similarity = monitor.get_digital_code(sequence=posi_code_this_seq, dictionary=position_code_used)
        word_code, similarity_1st, similarity_2nd = monitor.get_text_digital_code_this_sequence(features_this_seq, posi_idx, indices_ft,  centralize=True, output_zero=False)
        print(f'position indices:{posi_idx}')
        print(f'position similarity:{similarity}')
        print(f'position second similarity:{second_similarity}')
        print(f'word indices:{word_code}')
        print(f'text:{monitor.get_text(tokenizer, word_code)}')
        print(f'word similarity:{similarity_1st}')
        print(f'word second similarity:{similarity_2nd}')

