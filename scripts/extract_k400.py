import os
import subprocess
import json
from glob import glob
# from joblib import Parallel, delayed
from ipdb import set_trace
from tqdm import tqdm
import os.path as osp

# out_h = 256
# out_w = 256
# in_folder = 'data/UCF-101/'
# out_folder = 'data/ucf_{}x{}q5'.format(out_w,out_h)

# in_folder = '/UCF-101/'
# out_folder = '/ucf_{}x{}q5'.format(out_w,out_h)

# in_regs = ['/media/sda1/linhanxi/data/k400/nju_train_mp4/kinetics_400_train_10s_320p/*', '/media/sda1/linhanxi/data/k400/official_val_mp4/val_256/*/*']
# out_folder = '/media/sda1/linhanxi/data/k400/k400_CMNsplit/'

# in_regs = ['/media/nvme1/linhanxi/data/k400/nju_train_mp4/kinetics_400_train_10s_320p/*', '/media/nvme1/linhanxi/data/k400/official_val_mp4/val_256/*/*']
# out_folder = '/media/nvme1/linhanxi/data/k400/k400_CMNsplit/'
out_folder = '/home/linhanxi/home_data/k400_CMNsplit_30fps/'
in_regs = ['/media/sda1/linhanxi/data/k400/nju_train_mp4/kinetics_400_train_10s_320p/*', '/media/sda1/linhanxi/data/k400/official_val_mp4/val_256/*/*']

splits_dir = ["data/kinetics100/data_splits/meta_train.txt", "data/kinetics100/data_splits/meta_val.txt", "data/kinetics100/data_splits/meta_test.txt", 'data/kinetics100/data_splits/trainclasses_val.list']
# splits_dir = ['data/kinetics100/data_splits/trainclasses_val.list']

'''get all local vid_pth'''
local_vids = [glob(reg) for reg in in_regs]
local_vid_pth = [vid_pth for List in local_vids for vid_pth in List]
valid_vids = [osp.splitext(osp.basename(vid_pth))[0] for vid_pth in local_vid_pth]

'''get all required vid id'''
# requred_ids = []
for split_pth in splits_dir:
    with open(split_pth.replace('.txt', '_filtered.txt').replace('.list', '_filtered.list'), 'w') as new_f:
        valid_cnt, invalid_cnt = 0, 0
        for line in tqdm(open(split_pth)):
            # set_trace()
            if line.strip().split('/')[-1].__len__()!=11:
                # 11 is the regular length of youtuve vid id
                vid_id = line.strip().split('/')[-1][:-14]
            else:
                vid_id = line.strip().split('/')[-1]
            # set_trace()
            try:
                ind = valid_vids.index(vid_id)
                this_vid_pth = local_vid_pth[ind]
                valid_cnt += 1
                new_f.write(line)
                out_dir = osp.join(out_folder, vid_id)
                if not osp.exists(out_dir):
                    os.makedirs(out_dir)
                out_wc = osp.join(out_folder, vid_id, '%08d.jpg')
                cmd = ['ffmpeg', '-loglevel', 'error', '-i', this_vid_pth, '-q:v', '5', '-vf', "fps=30", out_wc]
                subprocess.call(cmd)
            except:
                invalid_cnt += 1
        print('{} vids valid, {} vids invalid for {}'.format(valid_cnt, invalid_cnt, split_pth))
    



# def run_cmd(cmd):
#     try:
#         os.mkdir(cmd[1])
#         subprocess.call(cmd[0])
#     except:
#         pass

# try:
#     os.mkdir(out_folder)
# except:
#     pass

# '''
# 3 for loops, first for splits, second for classes, third for vids
# '''
# for fn in glob(wc):
#     classes = []
#     vids = []

#     print(fn)
#     if "train" in fn:
#         cur_split = "train"
#     elif "val" in fn:
#         cur_split = "val"
#     elif "test" in fn:
#         cur_split = "test"

#     with open(fn, "r") as f:
#         data = f.readlines()
#         # 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01'
#         c = [x.split(os.sep)[-2].strip() for x in data] # class name
#         v = [x.split(os.sep)[-1].strip() for x in data] # video name
#         vids.extend(v)
#         classes.extend(c)

#     try:
#         os.mkdir(os.path.join(out_folder, cur_split))
#     except:
#         pass

#     for c in list(set(classes)):
#         try:
#             os.mkdir(os.path.join(out_folder, cur_split, c))
#         except:
#             pass

#     cmds = []
#     for v, c in zip(vids, classes):
#         source_vid = os.path.join(in_folder, c, "{}.avi".format(v))
#         extract_dir = os.path.join(out_folder, cur_split, c, v)

#         if os.path.exists(extract_dir):
#             continue

#         out_wc = os.path.join(extract_dir, '%08d.jpg')

#         print(source_vid, out_wc)

#         scale_string = 'scale={}:{}'.format(out_w, out_h)
#         os.mkdir(extract_dir)
#         try:
#             cmd = ['ffmpeg', '-i', source_vid, '-vf', scale_string, '-q:v', '5', out_wc]

#             cmds.append((cmd, extract_dir))
#             subprocess.call(cmd)
#         except:
#             pass
#     #Parallel(n_jobs=8, require='sharedmem')(delayed(run_cmd)(cmds[i]) for i in range(0, len(cmds)))