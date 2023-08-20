import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, AutoConfig
from edit_bert_classification import bert_backdoor_initialization
from data import get_dataloader, load_text_dataset
from torch.optim import SGD
from train import text_train, text_evaluation
import torch


def train_bert_classifier():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    use_backdoor_initialization = True
    learning_rate = 1e-4
    learning_rate_probe = 0.3
    dataset_name = 'trec'
    max_len = 48
    num_epochs = 5
    batch_size = 16
    device = 'cpu'
    is_simplify_arch = True
    save_path = './weights/bert_v1'

    train_dataset, test_dataset, num_classes = load_text_dataset(dataset=dataset_name, tokenizer=tokenizer, max_len=max_len)
    train_dataloader, test_dataloader = get_dataloader(train_dataset, ds1=test_dataset, batch_size=batch_size, num_workers=2)

    config = AutoConfig.from_pretrained('bert-base-uncased')

    if is_simplify_arch:
        config.hidden_act = 'relu'
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
    config.num_labels = num_classes
    config.output_attentions = True
    config.output_hidden_states = True
    classifier = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='bert-base-uncased', config=config)
    if is_simplify_arch:
        classifier.bert.pooler.activation = nn.ReLU()

    bert_monitor = None
    if use_backdoor_initialization:
        args = {}
        args['embedding_multipliers'] = {'features':1.0, 'position':1.0, 'word':1.0}
        args['embedding_large_constant'] = 1e5
        args['feature_syn_large_constant'] = 1e5
        args['signal_multiplier'] = 1.0
        args['bait_posi_multiplier'] = 0.1
        args['seq_signal_mutliplier'] = 1.0
        args['backdoor_mlp_large_constant'] = 1e5
        args['backdoor_multiplier'] = 1.0
        args['activation_signal_bound'] = 5.0
        args['limiter_large_constant'] = 1e5
        args['noise_threshold'] = 0.5
        args['last_activation_multiplier'] = 100.0
        args['features_add'] += 10.0
        bert_monitor = bert_backdoor_initialization(classifier, train_dataloader, args)

    optimizer = SGD([{'params': classifier.bert.parameters(), 'lr': learning_rate}, {'params': classifier.classifier.parameters(), 'lr': learning_rate_probe}])

    classifier = classifier.to(device)

    for j in range(num_epochs):
        print(f'Epoch: {j}')
        text_train(classifier, train_dataloader=train_dataloader, optimizer=optimizer, device=device)
        acc, val_loss = text_evaluation(classifier, evaluation_dataloader=test_dataloader, device=device)
        print(f'Validation ACC:{acc}, LOSS:{val_loss}')

    if bert_monitor is not None:
        torch.save(bert_monitor, f'{save_path}_{bert_monitor}.pth')
    torch.save(classifier.state_dict(), f'{save_path}_wt.pth')


if __name__ == '__main__':
    train_bert_classifier()