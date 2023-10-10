import os
import os.path as osp
import shutil
from ipdb import set_trace
import numpy as np


def join_multiple_txts(txt_pths):
    if isinstance(txt_pths, (list, np.ndarray)):
        for txt_pth in txt_pths:
            if osp.isfile(txt_pth):
                lines = open(txt_pth)
            else:
                set_trace()
                print('join_multiple_txts: input is not file, treated as string')
                lines = [txt_pth]
            for line in lines:
                yield line.strip()#.strip('\n').strip('\\n')
    else:
        for line in open(txt_pths):
            yield line.strip()#.strip('\n').strip('\\n')

            
def backup_files(filelist, dst_dir):
    if not osp.exists(dst_dir):
        raise ValueError(f'{dst_dir} not exists!')

    for file in filelist:
        basename = osp.basename(file)
        if osp.isfile(file):
            shutil.copy(file, dst=osp.join(dst_dir,basename))
        elif  osp.isdir(file):
            shutil.copytree(file, dst=osp.join(dst_dir,basename))
        else:
            raise ValueError(f'{file}')

    print(f'backup {filelist} to {dst_dir} !')


def flops(model, input):
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, input)
    print(flops.total())
    set_trace()



if __name__=='__main__':
    backup_files(
        ['few-shot-video-classification/shells/30902/ablation/test_r2plus1d_w_knowledge_k400CMN_ablation.sh',
        'few-shot-video-classification/datasets',
        'few-shot-video-classification/models',
        'few-shot-video-classification/building_model_w_knowledge.py',
        ],
        '/media/hdd/sda1/linhanxi/git_repos/few-shot-action-recognition/tmp'
        )




