import os
import shutil

root = '/home/linhanxi/home_data'
root = '/media/sda1/linhanxi/data/CLIP_related/kinetics_CMN/batched_to_be_processed_visual'
dirs = os.listdir(root)
for dirr in dirs:
    # if '256x' not in dirr:
    #     shutil.rmtree(os.path.join(root, dirr))
    if dirr.endswith('.pt'):
        os.remove(os.path.join(root, dirr))