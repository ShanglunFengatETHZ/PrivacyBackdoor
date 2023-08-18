import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AutoConfig
from data import get_dataloader, load_text_dataset
from torch.optim import SGD
from train import text_train, text_evaluation


def train_bert_classifier():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    learning_rate = 1e-4
    learning_rate_probe = 0.3
    dataset_name = 'trec'
    max_len = 48
    num_epochs = 5
    batch_size = 16
    device = 'cpu'
    is_simplify_arch = True

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

    optimizer = SGD([{'params': classifier.bert.parameters(), 'lr': learning_rate}, {'params': classifier.classifier.parameters(), 'lr': learning_rate_probe}])

    classifier = classifier.to(device)

    for j in range(num_epochs):
        print(f'Epoch: {j}')
        text_train(classifier, train_dataloader=train_dataloader, optimizer=optimizer, device=device)
        acc, val_loss = text_evaluation(classifier, evaluation_dataloader=test_dataloader, device=device)
        print(f'Validation ACC:{acc}, LOSS:{val_loss}')


if __name__ == '__main__':
    train_bert_classifier()