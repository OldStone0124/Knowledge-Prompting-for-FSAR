import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from torch import cuda
from ipdb import set_trace
from seqeval.metrics import classification_report
import os
import os.path as osp
import argparse
from utils import setup_logger, AverageMeter
from tokenize_inputs_and_match_labels import get_all_tokens_and_ner_tags_from_doccano
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.sentence[index].strip().split()  
        word_labels = self.data.word_labels[index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                            #  is_pretokenized=True, 
                             is_split_into_words=True,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        # Word pieces that should be ignored have a label of -100 (which is the default ignore_index of PyTorch's CrossEntropyLoss).
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        ''' `en`coding["offset_mapping"][?]` is the character_level start_ind/end_ind of the ?th word of the sentence w.r.t. the original word.
        for example, 'Mohsen' <=> [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0), ...'''
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label, the token label of the first wordpiece is set to the true label, otherwise -100
            encoded_labels[idx] = labels[i]
            i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        # every val if of list<[token_ids]>, with length len(sentence)
        item['labels'] = torch.as_tensor(encoded_labels)
        # set_trace()
        return item

  def __len__(self):
        return self.len


# Defining the training function on the 80% of the dataset for tuning the bert model
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        ret = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss, tr_logits = ret.loss, ret.logits
        # set_trace()
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            # loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            ret = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss, eval_logits = ret.loss, ret.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            # eval_labels.extend(labels)
            # eval_preds.extend(predictions)
            eval_labels.append(labels)
            eval_preds.append(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    # labels = [ids_to_labels[id.item()] for id in eval_labels]
    # predictions = [ids_to_labels[id.item()] for id in eval_preds]
    labels = [[ids_to_labels[id.item()] for id in batch] for batch in eval_labels]
    predictions = [[ids_to_labels[id.item()] for id in batch] for batch in eval_preds]
    # set_trace()

    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


@torch.no_grad()
def inference(sentence):
    # "as you watch him Bend he's going to Bend and met his hips go down here jump up and back "
    # sentence = "@HuggingFace is a company based in New York, but is also has employees working in Paris"
    inputs = tokenizer(sentence.split(),
                        #  is_pretokenized=True, 
                        is_split_into_words=True,
                        return_offsets_mapping=True, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=MAX_LEN,
                        return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
        else:
            continue

    print(sentence.split())
    print(prediction)   


# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=wPYV2Ld6Tr5I
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='log_dir')
    parser.add_argument('annotation_pth', type=str, help='annotation_pth')
    opt = parser.parse_args()
    if not osp.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    dataset_name = 'diving'
    # dataset_name = 'demo'
    if dataset_name=='demo':
        # demo dataset: https://www.kaggle.com/datasets/namanj27/ner-dataset
        data = pd.read_csv("annotations/ner_datasetreference.csv", encoding='unicode_escape')
        data.head()
        data.count()
        print("Number of tags: {}".format(len(data.Tag.unique())))
        frequencies = data.Tag.value_counts()
        frequencies
        tags = {}
        for tag, count in zip(frequencies.index, frequencies):
            if tag != "O":
                if tag[2:5] not in tags.keys():
                    tags[tag[2:5]] = count
                else:
                    tags[tag[2:5]] += count
            continue

        print(sorted(tags.items(), key=lambda x: x[1], reverse=True))
        labels_to_ids = {k: v for v, k in enumerate(data.Tag.unique())}
        ids_to_labels = {v: k for v, k in enumerate(data.Tag.unique())}

        data = data.fillna(method='ffill')
        data.head()

        # let's create a new column called "sentence" which groups the words by sentence 
        data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
        # let's also create a new column called "word_labels" which groups the tags by sentence 
        data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
        data.head()

        data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
        data.head()
        len(data)
        print(data.iloc[41].sentence, data.iloc[41].word_labels)
        set_trace()
    elif dataset_name=='diving':
        data = get_all_tokens_and_ner_tags_from_doccano(opt.annotation_pth, isBIO=True)
        set_trace()
        data['word_labels'] = data['ner_tags'].transform(lambda x: ','.join(x))
        data['sentence'] = data['tokens'].transform(lambda x: ' '.join(x))
        data = data.drop('tokens',axis='columns').drop('ner_tags',axis='columns')
        unique_entities = ['O', 'B-part_level_proposal', 'B-instance_level_proposal', 'I-part_level_proposal', 'I-instance_level_proposal']
        labels_to_ids = {k: v for v, k in enumerate(unique_entities)}
        ids_to_labels = {v: k for v, k in enumerate(unique_entities)}
        # data
    # set_trace()
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 100
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    '''A tricky part of NER with BERT is that BERT relies on wordpiece tokenization, rather than word tokenization. 
    This means that we should also define the labels at the wordpiece-level, rather than the word-level!
    '''
    # set_trace()
    train_size = 0.8
    train_dataset = data.sample(frac=train_size,random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = dataset(test_dataset, tokenizer, MAX_LEN)
    training_set[0]
    # set_trace()
    for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"]), training_set[0]["labels"]):
        print('{0:10}  {1}'.format(token, label))
    # set_trace()

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    logger = setup_logger(
        "perf_logger",
        opt.log_dir,
        0,
        'log.txt'
    )
    best_metric = 0
    select_metric_avgF1_fn = lambda x: float(x.split()[-2])
    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        train(epoch)
        labels, predictions = valid(model, testing_loader)
        report = classification_report(labels, predictions)
        selected_metric = select_metric_avgF1_fn(report)
        logger.info(report)
        if selected_metric>best_metric:
            logger.info(f'best_metric: {best_metric}')
            best_metric = selected_metric
            model.save_pretrained(opt.log_dir)
            tokenizer.save_vocabulary(opt.log_dir)
            # set_trace()
        # set_trace()

    set_trace()