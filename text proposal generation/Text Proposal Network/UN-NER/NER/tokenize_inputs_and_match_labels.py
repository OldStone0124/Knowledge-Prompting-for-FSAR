from json.tool import main
import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
from ipdb import set_trace
from data_process import load_jsonl


# dataset = 'demo'
dataset = 'diving'
if dataset=='demo':
    label_list = ['O','B-MISC','I-MISC','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']
    label_encoding_dict = {'I-PRG': 2,'I-I-MISC': 2, 'I-OR': 6, 'O': 0, 'I-': 0, 'VMISC': 0, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8, 'B-MISC': 1, 'I-MISC': 2}

    task = "ner" 
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 16
elif dataset=='diving':
    # label_list = ['O','instance_level_proposal','part_level_proposal']
    label_list = ['O', 'B-part_level_proposal', 'B-instance_level_proposal', 'I-part_level_proposal', 'I-instance_level_proposal']
    label_encoding_dict = {'instance_level_proposal': 2,'part_level_proposal': 1, 'O': 0}

    task = "ner" 
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 16
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def find_nth(haystack, needle, n):
    # https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def find_nth_space(string, start, direction, maxx):
    assert direction in [-1, 1]
    ret_pos = start-1
    pointer = start
    while maxx>0 and (len(string)-1)>pointer>0:
        pointer += direction
        if string[pointer] == ' ':
            ret_pos = pointer
            maxx -= 1
    return ret_pos


def get_all_tokens_and_ner_tags_from_doccano(jsonl_pth, range_ins=None, isBIO=False):
    dict_for_df = {"tokens":[], "ner_tags":[]}
    loaded = load_jsonl(jsonl_pth, check_label_empty=True)
    set_trace()
    for i,this_loaded in enumerate(loaded):
        if range_ins is not None and i not in range_ins:
            continue
        text, label = this_loaded['data'], this_loaded['label']
        for this_label in label:
            start_ind, end_ind, entity_type = this_label
            entity_text = text[start_ind:end_ind].strip()
            if isBIO:
                replaced_text = " ".join(['B-'+entity_type] + ['I-'+entity_type]*(len(entity_text.split())-1))
            else:
                replaced_text = " ".join([entity_type]*len(entity_text.split()))
            if '.' in text:
                # manual subtitles
                sent_start, sent_end = text.rfind('.', 0, start_ind), text.find('.', end_ind, len(text))
                sampled_sent = text[sent_start+1:sent_end]
            else:
                # auto-gen subtitles
                span_window = 4
                sent_start, sent_end = find_nth_space(text, start_ind, -1, span_window), find_nth_space(text, end_ind, 1, span_window)
                sampled_sent = text[sent_start+1:sent_end]
            transformed_sampled_sent = sampled_sent.replace(entity_text, replaced_text)
            # set_trace()
            tokenized_transformed_sampled_sent_sp = [ele if ele in label_list else 'O' for ele in transformed_sampled_sent.split()]
            dict_for_df['tokens'].append(sampled_sent.split())
            dict_for_df['ner_tags'].append(tokenized_transformed_sampled_sent_sp)
    # set_trace()
    assert len(dict_for_df['tokens'])==len(dict_for_df['ner_tags'])
    return pd.DataFrame(dict_for_df, index=range(len(dict_for_df['tokens'])))


def get_all_tokens_and_ner_tags(directory):
    data = [get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]
    # data[?]['ner_tags'][?] is list<nertag>
    # cated = pd.concat(data)
    # set_trace()
    return pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
    
def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list] 
        # set_trace()
        # tokens and entities are of same length. tokens[?] is the words of a sentence. entities[?] is the token labels of a sentence. 
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})


def get_token_dataset_from_doccano(annp_pth, train_ratio):
    assert 0<train_ratio<1
    length = len(open(annp_pth).readlines())
    break_pos = int(np.round(length*train_ratio))
    train_range = range(break_pos)
    val_range = range(break_pos, length)
    train_df = get_all_tokens_and_ner_tags_from_doccano(annp_pth, train_range)
    test_df = get_all_tokens_and_ner_tags_from_doccano(annp_pth, val_range)
    # set_trace()
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (train_dataset, test_dataset)


def get_un_token_dataset(train_directory, test_directory):
    train_df = get_all_tokens_and_ner_tags(train_directory)
    test_df = get_all_tokens_and_ner_tags(test_directory)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (train_dataset, test_dataset)


def tokenize_and_align_labels(examples):
    # **align token labels and tokenized_ids, because the tokenizetion may split the words up and some special tokens like [EOS] has been add.**
    # tokenize these datasets to convert the tokens and labels into numeric form for our BERT model.
    # Each token has now been transformed into a numeric representation and each label has been mapped to those used in upstream training
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        # for example, label==['O', 'O', 'O', 'instance_level_proposal', 'instance_level_proposal', 'instance_level_proposal', 'instance_level_proposal', 'instance_level_proposal']
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # set_trace()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    

if __name__=='__main__':
    # get_all_tokens_and_ner_tags('./tagged-test/')
    # df = get_tokens_and_ner_tags('./tagged-test/ID01939_2015_Canada.txt')
    if dataset=='demo':
        train_dataset, test_dataset = get_un_token_dataset('./tagged-training/', './tagged-test/')
    elif dataset=='diving':
        train_dataset, test_dataset = get_token_dataset_from_doccano("annotations/4_20220427.jsonl", 0.8)
    # set_trace()

    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)
    # set_trace()


    # **train_ner_model.py**
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    args = TrainingArguments(
        f"test-{task}",
        report_to='none',
        evaluation_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=1e-5,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")



    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=test_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model('un-ner.model')