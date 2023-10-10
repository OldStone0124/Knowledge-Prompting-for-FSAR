import os,sys
import os.path as osp
from glob import glob
from ipdb import set_trace
import collections
import random

def get_ids(txt_pth, get_class_func):
    ret = collections.OrderedDict()
    for line in open(txt_pth):
        line = line.strip()
        cls_name = get_class_func(line)
        if cls_name not in ret:
            ret[cls_name] = [line+'.avi']
        else:
            ret[cls_name].append(line+'.avi')
    return ret, ret.keys()

def get_candidates_ids(dirr, class_names):
    cands = collections.OrderedDict()
    for cls_name in class_names:
        cls_regex = osp.join(dirr, cls_name, '*')
        cands[cls_name] = glob(cls_regex)
    return cands

def minus(tobe_substracted, to_substract):
    new = collections.OrderedDict()
    assert tobe_substracted.keys()==to_substract.keys()
    for key in tobe_substracted.keys():
        new[key] = set(tobe_substracted[key]) - set(to_substract[key])
    return new


if __name__=='__main__':
    txt_pth, get_class_func = 'few-shot-video-classification/data/hmdb_ARN/train.txt', lambda x: x.split('/')[0]
    ret, class_names = get_ids(txt_pth, get_class_func)
    dirr = '/media/sda1/linhanxi/HMDB51/frame/'
    cands = get_candidates_ids(dirr, class_names)
    trainval = minus(cands, ret)
    max_select_num = min([len(trainval[k]) for k in trainval])
    dst_txt_pth = 'few-shot-video-classification/data/hmdb_ARN/trainval.txt'
    with open(dst_txt_pth, 'w') as f:
        for key in trainval.keys():
            f.write('\n'.join(random.sample(list(trainval['sword']), max_select_num))+'\n')
            f.flush()
    set_trace()

