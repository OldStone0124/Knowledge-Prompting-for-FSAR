import os
import os.path as osp
from pickle import FALSE
from ipdb import set_trace
import torch.nn as nn
import transformers
import torch.nn.functional as F
from tqdm import tqdm
import torch
from datetime import datetime
import pandas as pd

import nltk
from transformers.models import bert
from lin_utils import join_multiple_txts
import random
import numpy as np
from mid_level_action.utils import AverageMeter

'''
combine subjects, verbs and nouns'''
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def truncate(iterable, batch_size, drop_last=False):
    batch = []
    for idx in iterable:
        batch.append(idx)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0 and not drop_last:
        yield batch

def pprint(tuples):
    print('\n'.join([ele.__str__() for ele in tuples]))

def add_humans(text:str):
    return f'''Human's {text}'''

def preprocess_noun_txt(txt_pth:str):
    # preprocess_noun_txt('mid_level_action/obj_set.txt')
    # nouns = [ele.split('.')[0] for ele in open(txt_pth)]
    # nouns = [' '.join(ele.split('_')) for ele in nouns]
    nouns = [ele.split(' ')[-1].strip() for ele in open(txt_pth)]
    return nouns

def simple_instance_level_action_text_gen(label_txt:str, LM_examine:bool=False, MLM_thresh=1e-5, debug=False):
    name = osp.splitext(osp.basename(label_txt))[0]
    label_name_read = pd.read_csv(label_txt)
    label_texts = label_name_read.name.to_numpy()
    text_candidates = []
    for label_text in label_texts:
        text_candidates.append('Human is ' + label_text + '.')
    with open('mid_level_action/simple_mid_ins_level_proposals_{}_{}.txt'.format(name, datetime.now().strftime('%Y_%m%d_%Hh_%Mm_%Ss')) ,'w') as f:
        f.write('\n'.join(text_candidates))


@torch.cuda.amp.autocast()
@torch.no_grad()
def instance_level_action_text_gen(label_txt:str, LM_examine:bool=False, MLM_thresh=1e-5, debug=False):
    # **obtain verb_set, noun_set**
    verb_set, noun_set = set(), set()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    label_name_read = pd.read_csv(label_txt)
    label_texts = label_name_read.name.to_numpy()
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    pos_tags = []
    for label_text in label_texts:
        tokenized = nltk.word_tokenize(label_text)
        pos_tag=nltk.pos_tag(tokenized)
        pos_tags.append(pos_tag)
        # MVP ver, only consider single noun and single verb
        if len(pos_tag)==1:
            if pos_tag[0][1] in ('NN', 'NNS'):
                verb_set.add((pos_tag[0][0], 'v'))
            elif pos_tag[0][1]=='VBG':
                # verb_set.add(lemmatizer.lemmatize(pos_tag[0][0], pos='v'), 'v')
                verb_set.add((pos_tag[0][0], 'v'))
            else:
                set_trace()
                raise NotImplementedError
        elif len(pos_tag)>1:
            for pos,item in enumerate(pos_tag):
                # find the first 'NN'
                # *MVP ver, may lose information*
                if item[1] not in ('NN', 'NNS'):
                    continue
                else:
                    noun_set.add(item[0])
                    break
            if item[1] not in ('NN', 'NNS'):
                # no 'NN' is found, which means 'v'
                verb_phrase = ' '.join([ele[0] for ele in pos_tag])
                verb_set.add((verb_phrase, 'v'))
            else:
                if pos==0:
                    # case: air drumming
                    verb_phrase = ' '.join([ele[0] for ele in pos_tag])
                    verb_set.add((verb_phrase, 'v'))
                else:
                    # if 'vn', get the phrase before the noun
                    verb_phrase = ' '.join([ele[0] for ele in pos_tag[:pos]])
                    verb_set.add((verb_phrase, 'vn'))
    # set_trace()

    # **combine and examine**
    print('start mlm exaiming for proposals')
    text_candidates = []
    deprecated_cnt = 0
    if LM_examine:
        bert_mlm = BERT_MLM(MLM_thresh, debug=debug)
    for verb_item in tqdm(verb_set):
        verb, is_vn = verb_item
        if is_vn=='vn':
            verb = verb + ' {}'
            if LM_examine:
                masked_text_candidate = [' '.join(['Human is', verb.format('[MASK]')]) +'.',
                # ' '.join(['Human is', verb.replace('{}', 'a {}').format('[MASK]')]) +'.',
                # ' '.join(['Human is', verb.replace('{}', 'an {}').format('[MASK]')]) +'.',
                ]
                probs, top5 = bert_mlm.predict_one_word(masked_text_candidate)
            else:
                raise NotImplementedError
            for noun in noun_set:
                if LM_examine:
                    is_kept = bert_mlm.examine_one_word(noun)
                    # if noun_prob>MLM_thresh:
                    if is_kept:
                        text_candidate = ' '.join(['Human is', verb.format(noun)]) +'.'
                        text_candidates.append(text_candidate)
                    else:
                        deprecated_cnt += 1
        elif is_vn=='v':
            text_candidates.append(' '.join(['Human is', verb]) +'.')
        else: 
            raise ValueError
    print('deprecated:{}: ; preserved{}'.format(deprecated_cnt, len(text_candidates)))
    with open('mid_level_action/mid_ins_level_proposals_{}.txt'.format(datetime.now().strftime('%Y_%m%d_%Hh_%Mm_%Ss')) ,'w') as f:
        f.write('\n'.join(text_candidates))
    return text_candidates



@torch.cuda.amp.autocast()
@torch.no_grad()
def instance_level_action_text_gen_handcraft(label_txt:str, debug=False):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]

    # **obtain verb_set, noun_set**
    verb_set, noun_set = set(), set()
    label_name_read = pd.read_csv(label_txt)
    label_texts = label_name_read.name.to_numpy()
    text_candidates = []
    for tmpl in text_aug:
        for text in label_texts:
            text_candidates.append(tmpl.format(text))

    with open('mid_level_action/mid_ins_level_proposals_handcraft_{}.txt'.format(datetime.now().strftime('%Y_%m%d_%Hh_%Mm_%Ss')) ,'w') as f:
        f.write('\n'.join(text_candidates))
    return text_candidates


def domain_specific_noun_gen(label_txt:str, LM_examine:bool=False, MLM_thresh=1e-5, debug=False):
    # **obtain noun_set**
    noun_set = set()
    label_name_read = pd.read_csv(label_txt)
    label_texts = label_name_read.name.to_numpy()
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    pos_tags = []
    for label_text in label_texts:
        tokenized = nltk.word_tokenize(label_text)
        pos_tag=nltk.pos_tag(tokenized)
        pos_tags.append(pos_tag)
        # MVP ver, only consider single noun and single verb
        if len(pos_tag)==1:
            pass
        elif len(pos_tag)>1:
            for pos,item in enumerate(pos_tag):
                # find the first 'NN'
                # *MVP ver, may lose information*
                if item[1] not in ('NN', 'NNS'):
                    continue
                else:
                    noun_set.add(item[0])
                    break
    with open('mid_level_action/{}_specific_nouns_{}.txt'.format(osp.splitext(osp.basename(label_txt))[0], datetime.now().strftime('%Y_%m%d_%Hh_%Mm_%Ss')) ,'w') as f:
        f.write('\n'.join(noun_set))
    return noun_set


def intra_parts_interation_gen():
    return


class BERT_MLM():
    def __init__(self, examine_thresh, debug=False) -> None:
        # https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        # BertForPreTraining, BertForSequenceClassification
        self.examine_thresh = examine_thresh
        self.debug = debug
    
    def predict_one_word(self, masked_text_candidate):
        if self.debug:
            # noun = 'fashion'
            masked_text_candidate = "Hello I'm a [MASK] model."
            noun = 'dog'
            # masked_text_candidate = "Human's head kiss the [MASK]."
        encoded_input = self.tokenizer(masked_text_candidate, return_tensors='pt').to(device)
        # set_trace()
        mask_pos = (encoded_input['input_ids'][0]==self.tokenizer.mask_token_id).nonzero().item()
        # BERT inference
        output = self.model(**encoded_input)
        logits_mlm = output.logits[:,mask_pos,:]
        probs = F.softmax(logits_mlm, dim=-1).max(0)[0]
        top5 = [self.tokenizer.decode(idx) for idx in probs.topk(5)[1]]
        if self.debug:
            print(top5)
            probs[self.tokenizer.vocab['dog']]
            set_trace()
        self.probs, self.top5 = probs, top5
        return probs, top5

    def examine_one_word(self, noun):
        if noun in self.tokenizer.vocab:
            noun_token_id = self.tokenizer.vocab[noun]
            noun_prob = self.probs[noun_token_id]
            # print(noun, noun_prob)
            # set_trace()
            return noun_prob>self.examine_thresh
        else:
            # *warning: may lose valid proposals*
            return False


@torch.cuda.amp.autocast()
@torch.no_grad()
def part_level_action_text_gen(sub_verb_txt:str, noun_txt:str, LM_examine:bool=False, MLM_thresh=1e-5, debug=False, extra_nouns_pths=None):
    # part_level_action_text_gen('mid_level_action/Part_State_20211217.txt', 'mid_level_action/obj_set.txt)
    if LM_examine:
        bert_mlm = BERT_MLM(MLM_thresh, debug=debug)
        '''
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        '''
    nouns = preprocess_noun_txt(noun_txt)
    if extra_nouns_pths:
        extra_nouns = set([ele for ele in join_multiple_txts(extra_nouns_pths)])
        nouns = set(nouns)|extra_nouns

    text_candidates = []
    deprecated_cnt = 0
    for line in tqdm(open(sub_verb_txt)):
        splited = line.strip().split(',')
        subject, verb, is_vn = splited
        if is_vn=='vn':
            # deprecated_cnt = 0
            if LM_examine:
                '''
                if debug:
                    # noun = 'fashion'
                    # masked_text_candidate = "Hello I'm a [MASK] model."
                    noun = 'dog'
                    masked_text_candidate = "Human's head kiss the [MASK]."
                else:
                    masked_text_candidate = [' '.join([add_humans(subject), verb.replace('{}', 'the {}').format('[MASK]')]) +'.',
                    # ' '.join([add_humans(subject), verb.replace('{}', 'a {}').format('[MASK]')]) +'.',
                    # ' '.join([add_humans(subject), verb.replace('{}', 'an {}').format('[MASK]')]) +'.',
                    ]
                encoded_input = tokenizer(masked_text_candidate, return_tensors='pt').to(device)
                # set_trace()
                mask_pos = (encoded_input['input_ids'][0]==tokenizer.mask_token_id).nonzero().item()
                # BERT inference
                output = model(**encoded_input)
                logits_mlm = output.logits[:,mask_pos,:]
                probs = F.softmax(logits_mlm, dim=-1).max(0)[0]
                top5 = [tokenizer.decode(idx) for idx in probs.topk(5)[1]]
                if debug:
                    print(top5)
                    probs[tokenizer.vocab['dog']]
                    set_trace()
                '''
                masked_text_candidate = [' '.join([add_humans(subject), verb.replace('{}', 'the {}').format('[MASK]')]) +'.',
                # ' '.join([add_humans(subject), verb.replace('{}', 'a {}').format('[MASK]')]) +'.',
                # ' '.join([add_humans(subject), verb.replace('{}', 'an {}').format('[MASK]')]) +'.',
                ]
                probs, top5 = bert_mlm.predict_one_word(masked_text_candidate)
            for noun in nouns:
                # if LM_examine and noun in tokenizer.vocab:
                if LM_examine:
                    '''
                    noun_token_id = tokenizer.vocab[noun]
                    noun_prob = probs[noun_token_id]
                    # print(noun, noun_prob)
                    # set_trace()
                    '''
                    is_kept = bert_mlm.examine_one_word(noun)
                    # if noun_prob>MLM_thresh:
                    if is_kept:
                        text_candidate = ' '.join([add_humans(subject), verb.replace('{}', 'the {}').format(noun)]) +'.'
                        text_candidates.append(text_candidate)
                    else:
                        deprecated_cnt += 1
                else:
                    # set_trace()
                    continue
                    # **noun phrase longer than 2 words are deprecated**
                    text_candidates.append(text_candidate)
        elif is_vn=='v':
            text_candidates.append(' '.join([add_humans(subject), verb]) +'.')
        else: 
            raise ValueError

    print('deprecated:{}: ; preserved{}'.format(deprecated_cnt, len(text_candidates)))
    # set_trace()
    with open('mid_level_action/mid_level_act_proposals_{}.txt'.format(datetime.now().strftime('%Y_%m%d_%Hh_%Mm_%Ss')) ,'w') as f:
        f.write('\n'.join(text_candidates))
    return text_candidates


@torch.cuda.amp.autocast()
@torch.no_grad()
def part_level_action_text_gen_v1(sub_verb_txt:str, noun_txt:str, LM_examine:bool=False, MLM_thresh=1e-5, debug=False):
    # part_level_action_text_gen('mid_level_action/Part_State_20211217.txt', 'mid_level_action/obj_set.txt)
    if LM_examine:
        # https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        # set_trace()

    nouns = preprocess_noun_txt(noun_txt)
    text_candidates = []
    sub_vn_candidates = []
    for line in tqdm(open(sub_verb_txt)):
        splited = line.strip().split(',')
        subject, verb, is_vn = splited
        if is_vn=='vn':
            deprecated_cnt = 0
            if LM_examine:
                if debug:
                    # noun = 'fashion'
                    # masked_text_candidate = "Hello I'm a [MASK] model."
                    noun = 'dog'
                    masked_text_candidate = "Human's head kiss the [MASK]."
                else:
                    masked_text_candidate = [' '.join([add_humans(subject), verb.replace('{}', 'the {}').format('[MASK]')]) +'.',
                    # ' '.join([add_humans(subject), verb.replace('{}', 'a {}').format('[MASK]')]) +'.',
                    # ' '.join([add_humans(subject), verb.replace('{}', 'an {}').format('[MASK]')]) +'.',
                    ]
                encoded_input = tokenizer(masked_text_candidate, return_tensors='pt').to(device)
                # set_trace()
                mask_pos = (encoded_input['input_ids'][0]==tokenizer.mask_token_id).nonzero().item()
                # BERT inference
                output = model(**encoded_input)
                logits_mlm = output.logits[:,mask_pos,:]
                probs = F.softmax(logits_mlm, dim=-1).max(0)[0]
                top5 = [tokenizer.decode(idx) for idx in probs.topk(5)[1]]
                if debug:
                    print(top5)
                    probs[tokenizer.vocab['dog']]
                    set_trace()
            for noun in nouns:
                text_candidate = ' '.join([add_humans(subject), verb.replace('{}', 'the {}').format(noun)]) +'.'
                if LM_examine and noun in tokenizer.vocab:
                    noun_token_id = tokenizer.vocab[noun]
                    noun_prob = probs[noun_token_id]
                    # print(noun, noun_prob)
                    # set_trace()
                    if noun_prob>MLM_thresh:
                        text_candidates.append(text_candidate)
                    else:
                        deprecated_cnt += 1
                else:
                    # set_trace()
                    continue
                    # **noun phrase longer than 2 words are deprecated**
                    text_candidates.append(text_candidate)
        elif is_vn=='v':
            text_candidates.append(' '.join([add_humans(subject), verb]) +'.')
        else: 
            raise ValueError

    print('deprecated:{}: ; preserved{}'.format(deprecated_cnt, len(text_candidates)))
    set_trace()

@torch.cuda.amp.autocast()
@torch.no_grad()
def part_level_action_text_gen_v2(sub_verb_txt:str, noun_txt:str, LM_examine:bool=False, MLM_thresh=1e-5, debug=False, demo=False, max_noun_len=3, batchsize=1024, balanced=True):
    '''v2: support multi-length nouns'''
    meter1 = AverageMeter()
    meter2 = AverageMeter()
    # part_level_action_text_gen('mid_level_action/Part_State_20211217.txt', 'mid_level_action/obj_set.txt)
    def decode_sent(indexed_tokens, prediction, tokenizer):
        predicted_index = torch.argmax(prediction[-1, :]).item()
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        return predicted_text
    def gen_next_token(model, tokenizer, text):
        texts = [text]
        indexed_tokens = [tokenizer.encode(text) for text in texts]
        tokens_tensor = torch.tensor(indexed_tokens)
        model.eval()
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        decoded_sent = decode_sent(indexed_tokens, prediction, tokenizer)
        return {'logits':predictions,  'decoded_sent':decoded_sent}

    #if LM_examine:
        # from transformers import GPT2Tokenizer
        # from transformers import GPT2LMHeadModel
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        # if demo:
        #     texts = ['''Human's foot stand on the''', '''Human's leg is close with the''', '''Human's hip sit on the''']
        #     indexed_tokens = [tokenizer.encode(text) for text in texts]
        #     tokens_tensor = torch.tensor(indexed_tokens)
        #     with torch.no_grad():
        #         outputs = model(tokens_tensor)
        #         predictions = outputs[0]
        #     for sent_indexed_tokens, sent_prediction in zip(indexed_tokens, predictions):
        #         decoded_sent = decode_sent(sent_indexed_tokens, sent_prediction, tokenizer)
        #         print(decoded_sent)
        #     return
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

    nouns = preprocess_noun_txt(noun_txt)
    tokenized_nouns = tokenizer(nouns, return_tensors='pt', padding=True)['input_ids']
    # https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
    ids2word_dict = {v:k for k,v in tokenizer.vocab.items()}
    original_nouns = nouns
    nouns = [' '.join(filter(lambda x:x not in ['[CLS]', '[SEP]', '[PAD]'], [ids2word_dict[iid] for iid in tokenized.tolist()])) for tokenized in tokenized_nouns]
    len2nounsDic = {}
    for noun in nouns:
        noun = noun.split()
        cur_len = len(noun)
        if cur_len>max_noun_len or cur_len<=1: continue
        if cur_len not in len2nounsDic:
            len2nounsDic[cur_len] = [noun]
        else:
            len2nounsDic[cur_len].append(noun)
    # set_trace()
    text_candidates = []
    deprecated_cnt = 0
    for line in tqdm(open(sub_verb_txt)):
        splited = line.strip().split(',')
        subject, verb, is_vn = splited
        if is_vn=='vn':
            if LM_examine:
                masked_text_candidate = []
                # single-char nouns
                masked_text_candidate.append(' '.join([add_humans(subject), verb.replace('{}', 'the {}').format('[MASK]')]) +'.',
                )
                # multi-char nouns
                mult_words_noun2batchIdx = {}
                for cur_len, cur_nouns in len2nounsDic.items():
                    for cur_noun in cur_nouns:
                        for i in range(1,cur_len):
                            fillin = ['[MASK]']*cur_len
                            for j in range(i):
                                fillin[j] = cur_noun[j]
                            this_text_candidate = ' '.join([add_humans(subject), verb.replace('{}', 'the {}').format(' '.join(fillin) +'.')])
                            masked_text_candidate.append(this_text_candidate)
                            cur_noun_str = ' '.join(cur_noun)
                            if cur_noun_str not in mult_words_noun2batchIdx:
                                mult_words_noun2batchIdx[cur_noun_str] = []
                                # store current index
                                mult_words_noun2batchIdx[cur_noun_str].append(len(masked_text_candidate)-1)
                            else:
                                # store current index
                                mult_words_noun2batchIdx[cur_noun_str].append(len(masked_text_candidate)-1)
                # set_trace()
                if debug:
                    # noun = 'fashion'
                    # masked_text_candidate = "Hello I'm a [MASK] model."
                    noun = 'dog'
                    masked_text_candidate = "Human's head kiss the [MASK]."
                else:
                    pass
                encoded_input = tokenizer(masked_text_candidate, return_tensors='pt', padding=True).to(device)
                # set_trace()
                # encoded_input['input_ids'].size() == torch.Size([num_cands, padded_length])
                mask_positions = torch.nonzero(encoded_input['input_ids']==tokenizer.mask_token_id)
                # **BERT inference**
                result = []
                # set_trace()
                for cur_input_ids in torch.split(encoded_input['input_ids'], [len(batch) for batch in truncate(range(encoded_input['input_ids'].size(0)), batchsize)]):
                    result.append(model(input_ids=cur_input_ids).logits.detach().cpu().float())
                
                logits = torch.cat(result, dim=0)
                del result
                # logits_mask.size()==(num_cands, vocab_size)
                # set_trace()
                logits_mask = logits[encoded_input['input_ids']==tokenizer.mask_token_id]
                probs_mask = F.softmax(logits_mask, dim=-1)
            for noun, orig_noun in zip(nouns, original_nouns):
                text_candidate = ' '.join([add_humans(subject), verb.replace('{}', 'the {}').format(orig_noun)]) +'.'
                if LM_examine:
                    if max_noun_len>=len(noun.split())>1:
                        batchIndices = mult_words_noun2batchIdx[noun]
                        # select_prob = 10 if balanced else 1
                        select_prob = 5 if balanced else 1
                        select_prob *= probs_mask[0][tokenizer.vocab[noun.split()[0]]]
                        for batchIdx, cur_word in zip(batchIndices, noun.split()[1:]):
                            probs = probs_mask[batchIdx]
                            noun_token_id = tokenizer.vocab[cur_word]
                            noun_prob = probs[noun_token_id]
                            select_prob *= noun_prob
                        # set_trace()
                        meter1.update(select_prob)
                        if select_prob>MLM_thresh:
                            text_candidates.append(text_candidate)
                            # set_trace()
                        else:
                            deprecated_cnt += 1
                    elif len(noun.split())==1 and noun in tokenizer.vocab:
                        probs = probs_mask[0]
                        noun_token_id = tokenizer.vocab[noun]
                        noun_prob = probs[noun_token_id]
                        # print(noun, noun_prob)
                        #set_trace()
                        meter2.update(noun_prob)
                        if noun_prob>MLM_thresh:
                            text_candidates.append(text_candidate)
                        else:
                            deprecated_cnt += 1

                else:
                    text_candidates.append(text_candidate)
        elif is_vn=='v':
            text_candidates.append(' '.join([add_humans(subject), verb]) +'.')
        else: 
            raise ValueError

    print('deprecated:{}: ; preserved{}'.format(deprecated_cnt, len(text_candidates)))
    with open('mid_level_action/mid_level_act_proposalsV2_{}.txt'.format(datetime.now().strftime('%Y_%m%d_%Hh_%Mm_%Ss')) ,'w') as f:
        f.write('\n'.join(text_candidates))
    set_trace()

if __name__=='__main__':
    mode = 'part_gen_v2'
    # mode = 'ins_gen_handcraft'
    # mode = 'part_gen_v1'
    if mode=='part_gen':
        texts = part_level_action_text_gen('mid_level_action/Part_State_20211217.txt', 'mid_level_action/obj_set.txt', LM_examine=True, MLM_thresh=2e-4, debug=False, extra_nouns_pths=['mid_level_action/kinetics_400_labels_specific_nouns_2021_1228_16h_27m_02s.txt'])
    elif mode=='part_gen_v1':
        texts = part_level_action_text_gen_v1('mid_level_action/Part_State_20211217.txt', 'mid_level_action/obj_set.txt', LM_examine=True, MLM_thresh=2e-4, debug=False)
    elif mode=='part_gen_v2':
        # actioin knowledge
        texts = part_level_action_text_gen_v2('mid_level_action/Part_State_20211217.txt', 'mid_level_action/obj_set2.txt', LM_examine=True, MLM_thresh=1e-4, debug=False)
        # texts = part_level_action_text_gen_v2('mid_level_action/Part_State_20211217.txt', 'mid_level_action/obj_set.txt', LM_examine=True, MLM_thresh=6e-4, debug=False)
    elif mode=='ins_gen_handcraft':
        texts = instance_level_action_text_gen_handcraft('lists/kinetics_400_labels.csv', debug=False)
    elif mode=='ins_gen':
        texts = instance_level_action_text_gen('lists/kinetics_400_labels.csv', LM_examine=True, MLM_thresh=2e-4, debug=False)
    elif mode=='simple_ins_gen':
        texts = simple_instance_level_action_text_gen('lists/hmdb51_labels.csv', LM_examine=True, MLM_thresh=2e-4, debug=False)
        # texts = simple_instance_level_action_text_gen('lists/kinetics_400_labels.csv', LM_examine=True, MLM_thresh=2e-4, debug=False)
    elif mode=='domain_specific_noun_gen':
        domain_specific_noun_gen('lists/kinetics_400_labels.csv', LM_examine=True, MLM_thresh=2e-4, debug=False)
    else:
        raise NotImplementedError