import os
import os.path as osp
import collections
from lin_utils import join_multiple_txts
from ipdb import set_trace


def mk_split(videofolder_pths, dst_train_pth, dst_trainval_pth, dst_val_pth, dst_test_pth, val_ratio, return_split):
    ret = collections.OrderedDict()
    for line in join_multiple_txts(videofolder_pths):
        vid, n_frms, labelInd = line.strip().split()

        if labelInd not in ret:
            ret[labelInd] = []
        else:
            ret[labelInd].append(vid)
    
    with open(dst_train_pth, 'w') as f1, open(dst_trainval_pth, 'w') as f2,  open(dst_val_pth, 'w') as f3, open(dst_test_pth, 'w') as f4:
        for labelInd, vids in ret.items():
            vids = [osp.join(labelInd, vid) for vid in vids]
            if return_split(labelInd)=='train':
                # set_trace()
                break_pos = int(len(vids)*val_ratio)
                f1.write('\n'.join(vids[break_pos:]) + '\n')
                f2.write('\n'.join(vids[:break_pos]) + '\n')
                f1.flush()
                f2.flush()
            elif return_split(labelInd)=='val':
                f3.write('\n'.join(vids) + '\n')
                f3.flush()
            elif return_split(labelInd)=='test':
                f4.write('\n'.join(vids) + '\n')
                f4.flush()
            else:
                raise


if __name__=='__main__':
    train_labelIndices = range(36)
    val_labelIndices = range(36, 42)
    test_labelIndices = range(42, 48)
    def return_split(labelInd):
        labelInd = int(labelInd)
        if labelInd in train_labelIndices:
            return 'train'
        elif labelInd in val_labelIndices:
            return 'val'
        elif labelInd in test_labelIndices:
            return 'test'
        else:
            raise
    videofolder_pths, dst_train_pth, dst_trainval_pth, dst_val_pth, dst_test_pth, val_ratio, return_split_fn = \
        ['few-shot-video-classification/data/diving48V2/train_videofolder.txt', 'few-shot-video-classification/data/diving48V2/val_videofolder.txt'], \
        'few-shot-video-classification/data/diving48V2/meta_train.txt', \
        'few-shot-video-classification/data/diving48V2/meta_trainval.txt', \
        'few-shot-video-classification/data/diving48V2/meta_val.txt', \
        'few-shot-video-classification/data/diving48V2/meta_test.txt', \
        0.18, \
        return_split
    mk_split(videofolder_pths, dst_train_pth, dst_trainval_pth, dst_val_pth, dst_test_pth, val_ratio, return_split)