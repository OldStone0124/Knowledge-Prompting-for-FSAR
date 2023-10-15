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
from glob import glob
from tqdm import tqdm
from utils import setup_logger, AverageMeter
from tokenize_inputs_and_match_labels import get_all_tokens_and_ner_tags_from_doccano
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


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

    # print(sentence.split())
    # print(prediction)   
    return sentence.split(), prediction

def extract_proposal(sentence_sp, prediction):
    ret = {}
    type = None
    tem = None
    for orig_sent_traval, pred_travel in zip(sentence_sp, prediction):

        if type is None and pred_travel.startswith('B-'):
            type = pred_travel[2:]
            tem = [orig_sent_traval]
        elif type is not None: 
            if pred_travel== ('I-'+type):
                tem.append(orig_sent_traval)
            else:
                if len(tem)>1:
                    # interupt
                    this_proposal = " ".join(tem)
                    if type not in ret:
                        ret[type] = []
                    ret[type].append(this_proposal)
                # refresh
                type = None
                tem = None
        else:
            pass
    return ret


def traverse_across_sentence(text, window_span, overlap):
    assert 0<overlap<1
    sp = text.split()
    orig_length = len(sp)
    stride = int(window_span*overlap)
    nums_to_pad = (orig_length-window_span)%stride
    # pad
    sp.append(" "*nums_to_pad)
    for i in range(1+(len(sp)-window_span)//stride):
        sampled = " ".join(sp[i*stride:i*stride+window_span])
        yield sampled


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='log_dir')
    parser.add_argument('output_dir', type=str, help='output_dir')
    parser.add_argument('--subtitles_regex', type=str, nargs='+', help='regex')
    opt = parser.parse_args()

    if not osp.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    else:
        raise ValueError(f'{opt.output_dir}')

    MAX_LEN = 128
    unique_entities = ['O', 'B-part_level_proposal', 'B-instance_level_proposal', 'I-part_level_proposal', 'I-instance_level_proposal']
    labels_to_ids = {k: v for v, k in enumerate(unique_entities)}
    ids_to_labels = {v: k for v, k in enumerate(unique_entities)}
    
    model = BertForTokenClassification.from_pretrained(osp.join(opt.log_dir, ''), num_labels=len(labels_to_ids))
    model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained(osp.join(opt.log_dir, ''))

    # sentence_sp, prediction = inference("as you watch him Bend he's going to Bend and met his hips go down here jump up and back ")
    # sentence_sp, prediction = inference(" he's going to dive in headfirst but he's going to leave his hands right down his side ready 1 2 3")
    # extract_proposal(sentence_sp, prediction)
    # set_trace()
    all_extracted = {"part_level_proposal":[], "instance_level_proposal":[]}
    subtitles_pths = []
    for regex in opt.subtitles_regex:
        subtitles_pths.extend(glob(regex))

    #set_trace()
    blocked_list = ['wW3n48BScFo', 'Pd3kzYS16MA', 'ofKPIqMQ48s', 's7Pa_FpWdkI', 'SOh7vYDXYNU', 'nq5GfrV_n2Y', 'Jn6uCLbDppU']
    del_list = []
    for i,pth in enumerate(subtitles_pths):
        for blocked in blocked_list:
            if osp.basename(pth).startswith(blocked):
                del_list.append(i)
                break
    for todel in del_list[::-1]:
        subtitles_pths.pop(todel)

    #set_trace()
    for subtitle_pth in tqdm(subtitles_pths):
        text = open(subtitle_pth).read()
        if ',' not in text:
            for sampled_line in traverse_across_sentence(text, 15, 0.5):
                sentence_sp, prediction = inference(sampled_line)
                #set_trace()
                extracted = extract_proposal(sentence_sp, prediction)
                # print(np.array(([[a,b] for a,b in zip( sentence_sp, prediction)] )))
                # set_trace()
                for key in all_extracted.keys():
                    if key in extracted:
                        all_extracted[key].extend(extracted[key])
        else:
            for sentence in text.split('.'):
                for sampled_line in traverse_across_sentence(sentence, 15, 0.5):
                    sentence_sp, prediction = inference(sampled_line)
                    extracted = extract_proposal(sentence_sp, prediction)
                    # print(np.array(([[a,b] for a,b in zip( sentence_sp, prediction)] )))
                    # set_trace()
                    for key in all_extracted.keys():
                        if key in extracted:
                            all_extracted[key].extend(extracted[key])
    # set_trace()
    filters = []
    filter1 = lambda x:len(x)>1
    for k,v in all_extracted.items():
        f = open(osp.join(opt.output_dir, f'extracted_{k}'), 'w')
        filtered_v = list(filter(filter1, v))
        # combined_v = [" ".join(ele) for ele in filtered_v]
        combined_v = filtered_v
        # set_trace()

        set_v = list(set(combined_v))
        f.write('\n'.join(set_v))
        f.close()

    # set_trace()
