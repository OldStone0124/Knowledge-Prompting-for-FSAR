import os
import collections

def mk_split(txt_pth, dst_train_pth, dst_val_pth, val_ratio, cls_gen_func):
    ret = collections.OrderedDict()
    for line in open(txt_pth):
        line = line.strip()
        cls_name = cls_gen_func(line)
        if cls_name not in ret:
            ret[cls_name] = [line]
        else:
            ret[cls_name].append(line)
    with open(dst_train_pth, 'w') as f1, open(dst_val_pth, 'w') as f2:
        for key in ret.keys():
            break_pos = int(len(ret[key])*val_ratio)
            f1.write('\n'.join(ret[key][break_pos:]) + '\n')
            f2.write('\n'.join(ret[key][:break_pos]) + '\n')


if __name__=='__main__':
    txt_pth, dst_train_pth, dst_val_pth, val_ratio, cls_gen_func = \
    'few-shot-video-classification/data/hmdb_ARN/train.txt', \
    'few-shot-video-classification/data/hmdb_ARN/meta_train.txt', \
    'few-shot-video-classification/data/hmdb_ARN/meta_val.txt', \
    0.18, lambda x:x.split('/')[0]
    mk_split(txt_pth, dst_train_pth, dst_val_pth, val_ratio, cls_gen_func)




