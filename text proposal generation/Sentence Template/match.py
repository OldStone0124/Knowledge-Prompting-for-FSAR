from ipdb.__main__ import set_trace
import sys
import os
sys.path.remove(os.getcwd())

import clip
print(clip.__file__)
#set_trace()
sys.path.insert(0, os.getcwd())
# sys.path.append('')
import decord
import torch
from utils.Augmentation import *
import yaml
from dotmap import DotMap
from glob import glob
import os
import os.path as osp
from tqdm import tqdm
from mid_level_action.mp_dataset import mp_vid_load_dataset, mp_frms_load_dataset
from torch.utils.data import DataLoader
from datetime import datetime
from lin_utils import lin_logger
from lin_utils import join_multiple_txts
import time


def chunk(iterable, batch_size, drop_last=False):
    batch = []
    for idx in iterable:
        batch.append(idx)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0 and not drop_last:
        yield batch

def evenly_sample(vid_pth, num_segments):
    container = decord.VideoReader(vid_pth)
    num_frames = len(container)
    indices = [int((i+0.5) * num_frames // num_segments)
                for i in range(num_segments)]
    # RGB
    imgs = [Image.fromarray(container[p-1].asnumpy()).convert('RGB') for p in indices]
    return imgs

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.cuda.amp.autocast()
@torch.no_grad()
def extract_proposal_fea(proposal_txt, backbone, store_dir):
    if not osp.exists(store_dir):
        os.makedirs(store_dir)
    model, preprocess = clip.load(backbone, device=device, jit=False)
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    model.eval()
    bs = 4
    # proposals = [line.strip() for line in open(proposal_txt).readlines()]
    proposals = [line.strip() for line in join_multiple_txts(proposal_txt)]
    proposal_num = len(proposals)
    cnt = 1
    for batched in tqdm(chunk(proposals, bs)):
        # set_trace()
        text_tokens = clip.tokenize(batched).cuda()
        batched_fea = model.encode_text(text_tokens).cpu()
        for fea in batched_fea:
            torch.save(fea, osp.join(store_dir, f'{cnt}.pt'))
            cnt += 1

    return proposal_num

@torch.cuda.amp.autocast()
@torch.no_grad()
def extract_proposal_fea_v2(proposal_txt, backbone, store_dir, save_name):
    if not osp.exists(store_dir):
        os.makedirs(store_dir)
    model, preprocess = clip.load(backbone, device=device, jit=False)
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    model.eval()
    bs = 4
    # proposals = [line.strip() for line in open(proposal_txt).readlines()]
    proposals = [line.strip() for line in join_multiple_txts(proposal_txt)]
    proposal_num = len(proposals)
    fea_list = []
    start = time.time()
    for batched in tqdm(chunk(proposals, bs)):
        # set_trace()
        text_tokens = clip.tokenize(batched).cuda()
        batched_fea = model.encode_text(text_tokens).cpu()
        for fea in batched_fea:
            fea_list.append(fea)

    end = time.time()

    print(end - start)

    proposal_fea = torch.stack(fea_list, 0)
    torch.save(proposal_fea, osp.join(store_dir, f'{save_name}.pt'))
    return proposal_num


@torch.cuda.amp.autocast()
@torch.no_grad()
def extract_vids_fea(vid_list, specific=None):
    # '/home/linhanxi/git_repos/general-action-recog/lists/k4001/train_frame.txt'
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # model = torch.nn.DataParallel(model).to(device)
    model.eval()
    model = model.to(device)

    bs = 2
    num_segments = 8
    vid_list = [line.strip() for line in open(vid_list).readlines()]
    for line in tqdm(vid_list):
        if specific:
            if specific not in line:
                continue
        vid_pth = line.split()[0]
        vid_pth = glob(vid_pth.replace('/media/hdd/sdb1', '/media/nvme1')+'*')[0]
        imgs = evenly_sample(vid_pth, num_segments)

        transformed = [preprocess(img) for img in imgs]
        image_input = torch.tensor(np.stack(transformed)).to(device)
        # set_trace()

        feas = model.encode_image(image_input)
        sotre_name = (osp.splitext(vid_pth)[0]+'.pt').replace('/media/nvme1/linhanxi/data/k400', '/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32')
        if not osp.exists(osp.dirname(sotre_name)):
            os.makedirs(osp.dirname(sotre_name))
        torch.save(feas, sotre_name)
        print(vid_pth)


@torch.cuda.amp.autocast()
@torch.no_grad()
def extract_vids_fea_v2(vid_list):
    # '/home/linhanxi/git_repos/general-action-recog/lists/k4001/train_frame.txt'
    with open('configs/k400/k400_zero_shot.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # model = torch.nn.DataParallel(model).to(device)
    model.eval()
    model = model.to(device)
    transform_val = get_augmentation(False, config)

    bs = 8
    num_segments = 8
    vid_list = [line.strip() for line in open(vid_list).readlines()]
    for i, chunked in enumerate(tqdm(chunk(vid_list, bs))):
        batched_list = []
        for line in chunked:
            vid_pth = line.split()[0]
            vid_pth = glob(vid_pth.replace('/media/hdd/sdb1', '/media/nvme1')+'*')[0]
            container = decord.VideoReader(vid_pth)
            num_frames = len(container)
            indices = [int((i+0.5) * num_frames // num_segments)
                        for i in range(num_segments)]
            # RGB
            imgs = [Image.fromarray(container[p-1].asnumpy()).convert('RGB') for p in indices]

            transformed = [preprocess(img) for img in imgs]
            batched_list.extend(transformed)
        image_input = torch.tensor(np.stack(batched_list)).to(device)
        # set_trace()

        feas = model.encode_image(image_input)
        torch.save(feas, '/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/to_be_processed/{}'.format(f'{i}.pt'))
        print(i)


@torch.cuda.amp.autocast()
@torch.no_grad()
def extract_vids_fea_v3(vid_list, out_dir, backbone, specific=None, prefix_root=None, data_mode='video', overwrite=False):
    #set_trace()
    name = osp.splitext(osp.basename(vid_list))[0]
    def collate_func(List):
        return torch.cat(List, dim=0)
    # '/home/linhanxi/git_repos/general-action-recog/lists/k4001/train_frame.txt'
    model, preprocess = clip.load(backbone, device=device, jit=False)
    bs = 1
    num_segments = 32
    if data_mode=='video':
        dataset = mp_vid_load_dataset(vid_list, num_segments, preprocess)
    elif data_mode=='frame':
        dataset = mp_frms_load_dataset(vid_list, num_segments, preprocess, prefix_root)
        pass
    else:
        raise
    val_loader = DataLoader(dataset,batch_size=bs, num_workers=0,shuffle=False,pin_memory=False,drop_last=False, collate_fn=collate_func)
    # set_trace()
    # model = torch.nn.DataParallel(model).to(device)
    model.eval()
    model = model.to(device)

    vid_list = [line.strip() for line in open(vid_list).readlines()]
    for i,batched in enumerate(tqdm(val_loader)):
        store_dir = osp.join(
            out_dir,
            'fea_clip_{}/batched_to_be_processed_visual/{}'.format(backbone, name))
        if not overwrite:
            if osp.exists('{}/{}'.format(store_dir, f'{i}.pt')):
                continue
        batched = batched.to(device)
        # set_trace()
        '''
        batched.size()
        torch.Size([bs*8, 3, 224, 224])'''
        feas = model.encode_image(batched).cpu()
        if not osp.exists(store_dir):
            os.makedirs(store_dir)

        torch.save(feas, '{}/{}'.format(store_dir, f'{i}.pt'))
        print('write:{}'.format(f'{i}.pt'))


@torch.cuda.amp.autocast()
@torch.no_grad()
def extract_vids_fea_v5(regex, out_dir, backbone, num_segments, bs=16, data_mode='video', overwrite=False, tranform_func=None, id_tranform_func=None, batch_save=True, save_dir_suffix=''):
    # **extract according to the regex**
    # ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16']
    # set_trace()
    vid_list = glob(regex)
    def collate_func(List):
        return torch.cat([ele[0] for ele in List], dim=0), [ele[1] for ele in List]
        # return torch.cat(List, dim=0)
    # '/home/linhanxi/git_repos/general-action-recog/lists/k4001/train_frame.txt'
    model, preprocess = clip.load(backbone, device=device, jit=False)
    if data_mode=='video':
        dataset = mp_vid_load_dataset(vid_list, num_segments, preprocess)
    elif data_mode=='frame':
        dataset = mp_frms_load_dataset(vid_list, num_segments, preprocess, tranform_func=tranform_func, id_tranform_func=id_tranform_func, return_id=True)
    else:
        raise
    val_loader = DataLoader(dataset,batch_size=bs, num_workers=8,shuffle=False,pin_memory=False,drop_last=False, collate_fn=collate_func)

    # model = torch.nn.DataParallel(model).to(device)
    model.eval()
    model = model.to(device)

    # vid_list = [line.strip() for line in open(vid_list).readlines()]
    for i,batched in enumerate(tqdm(val_loader, leave=True)):
        # continue
        batched, iids = batched
        store_dir = osp.join(
            out_dir,
            'batched_to_be_processed_visual/fea_clip_{}{}/'.format(backbone.replace('/','-'), save_dir_suffix))

        batched = batched.to(device)
        # set_trace()
        '''
        batched.size()
        torch.Size([bs*8, 3, 224, 224])'''
        start_time = time.time()
        feas = model.encode_image(batched).cpu()
        print(time.time() - start_time)
        this_bs = feas.size(0)//num_segments
        feas = torch.split(feas, [num_segments]*this_bs,dim=0)
        # set_trace()
        
        if batch_save:
            torch.save(feas, osp.join(store_dir, f'{i}.pt'))
            print('write:{}'.format(f'{i}.pt'))
        else:
            for j, (fea,iid) in enumerate(zip(feas, iids)):
                filename = iid.split('/')[-2]
                iid = iid.split('/')[-1]
                iid = iid.split('.')[0]
                dst_pth = osp.join(store_dir, filename, f'{iid}.pt')
                dst_dir = osp.dirname(dst_pth)
                if not osp.exists(dst_dir):
                    os.makedirs(dst_dir)
                # starts from 1
                # **clone is necessary**
                if fea.size(0)==0:
                    raise
                if not overwrite:
                    if osp.exists(dst_pth):
                        continue
                # set_trace()
                torch.save(fea.clone(), dst_pth)
                print('write:{}'.format(dst_pth))


@torch.no_grad()
def combine(store_dir):
    store_pth = '/'.join(store_dir.split('/')[:-1]) + '.pt'
    L = []
    for pth in sorted(os.listdir(store_dir), key=lambda x:int(osp.splitext(x)[0])):
        # set_trace()
        loaded = torch.load(osp.join(store_dir, pth), map_location='cpu')
        L.append(loaded)
    torch.save(torch.stack(L), store_pth)
    return store_pth


@torch.no_grad()
class scaled_cosine_sim():
    def __init__(self, visual_arch):
        model, preprocess = clip.load(visual_arch, device='cpu')
        self.scale = model.logit_scale.detach().cpu().exp()
        del model, preprocess

    def __call__(self, aa, bb):
        if not isinstance(aa, torch.Tensor):
            aa=torch.tensor(aa)
            bb=torch.tensor(bb)
        aa /= aa.norm(dim=-1, keepdim=True)
        bb /= bb.norm(dim=-1, keepdim=True)
        sim_mat = (self.scale* aa @ bb.T)
        probs = torch.nn.functional.softmax(sim_mat, dim=-1)
        return probs


@torch.cuda.amp.autocast()
@torch.no_grad()
def match_one(vid_pth, proposal_txt=''):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    fea_pth = osp.join(vid_pth)
    if osp.exists(fea_pth):
        image_features = torch.load(fea_pth, map_location='cpu').to(device)
    else:
        raise
        model, clip_state_dict = clip.load("ViT-B/32", device=device, jit=False, tsm=False,
                                                    T=8, dropout=.0,
                                                    emb_dropout=.0)
        model = torch.nn.DataParallel(model).to(device)
        imgs = evenly_sample(vid_pth, 8)
    # text
    proposals_fea = torch.load('/media/nvme1/linhanxi/data/ActionKnowledge/proposal_20211219/combined.pt').to(device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    proposals_fea /= proposals_fea.norm(dim=-1, keepdim=True)
    sim_mat = (image_features @ proposals_fea.T).cpu()
    # (num_segments, num_proposals)
    set_trace()
    proposals=np.array(open('mid_level_action/mid_level_act_proposals_2021_1219_17h_29m_28s.txt').readlines())
    proposals[sim_mat[3,:].float().topk(5).indices]


@torch.cuda.amp.autocast()
@torch.no_grad()
def match_all(reg, proposal_fea_pth, replace_dict, visual_arch, overwrite=False):
    scaled_cosine_simer = scaled_cosine_sim(visual_arch)
    proposals_fea = torch.load(proposal_fea_pth).to(device)
    # set_trace()
    feas = glob(reg)
    for fea_pth in tqdm(feas):
        target_pth = fea_pth.replace(replace_dict['old_root'], replace_dict['new_root'])
        if not overwrite:
            if osp.exists(target_pth):
                print('found {} exists'.format(target_pth))
                continue

        loaded_v = torch.load(fea_pth).to(device)
        # match = loaded_v@proposals_fea.T
        match = scaled_cosine_simer(loaded_v, proposals_fea) # probs
        # set_trace()
        if not osp.exists(osp.dirname(target_pth)):
            os.makedirs(osp.dirname(target_pth))

        torch.save(match.float().cpu(), target_pth)
        print('write:{}'.format(target_pth))
    return replace_dict['new_root']
    

def probs2indices(batched_probs_dirr:str, num_segments:int, pth_to_save:str, nums_to_preserve:int):
    reg = osp.join(batched_probs_dirr, '*')
    batched_list = sorted(glob(reg), key=lambda x:int(osp.splitext(osp.basename(x))[0]))
    assert  torch.load(batched_list[0]).size(0) % num_segments == 0
    assert  torch.load(batched_list[-1]).size(0) % num_segments == 0
    num_samples = (len(batched_list)-1)*torch.load(batched_list[0]).size(0)//num_segments + 1*torch.load(batched_list[-1]).size(0)//num_segments
    num_proposals = torch.load(batched_list[0]).size(-1)
    indices, values  = torch.zeros(num_samples,num_segments,nums_to_preserve, dtype=torch.int32), torch.zeros(num_samples,num_segments,nums_to_preserve, dtype=torch.float32)
    obj_to_save = {'num_samples':num_samples, 'num_proposals':num_proposals}
    cnt = -1
    for batch_pth in tqdm(batched_list):
        loaded = torch.load(batch_pth)
        for each in torch.split(loaded, num_segments):
            cnt += 1
            assert cnt<=(num_samples-1)
            topk_return = each.topk(nums_to_preserve)
            indices[cnt] = topk_return.indices
            assert topk_return.indices.max().item() <= (num_proposals-1)
            values[cnt] = topk_return.values
    obj_to_save['indices'] = indices
    obj_to_save['values'] = values
    torch.save(obj_to_save, pth_to_save)


def new_match_4_new_proposals(proposal_txt, proposal_fea_cache_dir, match_result_cache_dir, vis_fea_reg, log_file, backbone, store_mode, setting={}, overwrite=False):
    assert store_mode in ['indices', 'probs']
    #set_trace()
    log_handle = lin_logger(log_file, overwrite=False)
    time = datetime.now().strftime('%Y_%m%d_%Hh_%Mm_%Ss')
    log_handle.write_and_print(time)
    log_handle.write_and_print('new_match_4_new_proposals called:')
    log_handle.write_and_print('start extract_proposal_fea:')
    extract_proposal_fea(proposal_txt, backbone, proposal_fea_cache_dir)

    log_handle.write_and_print('start combine:')
    combined_pth = combine(proposal_fea_cache_dir)

    vis_fea_dir = osp.join(*filter(lambda x:x!='*', vis_fea_reg.split('/')))
    log_handle.write_and_print('start match_all to {}:'.format(vis_fea_dir))
    matched_dir = match_all(
        vis_fea_reg, 
        combined_pth, 
        {'old_root':vis_fea_dir, 'new_root':match_result_cache_dir}, 
        backbone,
        overwrite=overwrite)

    # set_trace()
    log_handle.write_and_print('start remove proposal_cache_cache_dir:\n')
    os.system('rm -r {}'.format(proposal_fea_cache_dir))    

    # set_trace()

    if store_mode=='probs':
        unfolded_match_dir = setting['unfolded_match_dir']
        log_handle.write_and_print('start unfold to {}:'.format(unfolded_match_dir))
        from mid_level_action.unfold_batch import unfold_batch
        second_names = os.listdir(matched_dir)
        for second_name in second_names:
            unfold_batch(
                osp.join(matched_dir, second_name, '*'), 'dummy', setting['num_segments'], 
                {'old_root':matched_dir, 'new_root':unfolded_match_dir})
            
    elif store_mode=='indices':
        log_handle.write_and_print('start batched_probs2indices to {}:'.format(osp.join(setting['pth_to_save_dir'])))
        probs2indices(batched_probs_dirr=osp.join(matched_dir, 'train_frame'),
         num_segments=setting['num_segments'], 
         pth_to_save=osp.join(setting['pth_to_save_dir'], 'train.pt'),
         nums_to_preserve=setting['nums_to_preserve'])
        probs2indices(batched_probs_dirr=osp.join(matched_dir, 'val_frame'),
         num_segments=setting['num_segments'], 
         pth_to_save=osp.join(setting['pth_to_save_dir'], 'val.pt'),
         nums_to_preserve=setting['nums_to_preserve'])
    else:
        raise
    log_handle.write_and_print('start remove clip match cache dir:\n')
    os.system('rm -r {}'.format(matched_dir))    


if __name__=='__main__':
    mode = 'v5'
    #mode = 'new_match_4_new_proposals'
    # mode = 'probs2indices'
    if mode=='v':
        # extract_vids_fea('lists/k4001/train_frame.txt')
        # extract_vids_fea('lists/k4001/val_frame.txt', 'OCQSpPkCAWw')

        '''
        extract_vids_fea_v3('lists/k4001/train_frame.txt')
        extract_vids_fea_v3('lists/k4001/val_frame.txt')
        '''
        for txt_pth, prefix in zip(
            ['/media/sda1/shiyuheng/data/gym/annotations/gym99_frame.txt'], 
            ['/media/sda1/shiyuheng/data/gym/subaction_frames/']
        ):
            extract_vids_fea_v3(txt_pth, '/media/sda1/shiyuheng/actionKnowledgeXfewshot/few-shot-video-classification/data_CLIP_visual_fea/finegym/', 'ViT-B-16', prefix_root=prefix, data_mode='frame', overwrite=False)
    
    elif mode=='v5':# 提取视频特征
        regex = '/media/sda1/shiyuheng/data/backup-data/data/hmdb51/frame/*/*'
        tranform_func = lambda x: x
        id_tranform_func = lambda x: x.strip().split('/')[-1]
        extract_vids_fea_v5(regex, 
            '/home/shiyuheng/data/data_CLIP_visual_fea/test_time',
            'RN50', 32, 16, tranform_func=tranform_func, id_tranform_func=id_tranform_func,
            data_mode='frame', batch_save=False, save_dir_suffix='_unfolded_32frms'
        )
    
    elif mode=='t':
        extract_proposal_fea('mid_level_action/mid_level_act_proposals_2021_1219_17h_29m_28s.txt', 'RN50', store_dir = '/home/shiyuheng/ActionCLIP/textProposal')
        # extract_proposal_fea(['mid_level_action/mid_ins_level_proposals_2021_1226_22h_07m_16s.txt'], store_dir = '/home/shiyuheng/ActionCLIP/textProposal')
        # extract_proposal_fea(['mid_level_action/mid_level_act_proposals_2021_1228_21h_04m_35s.txt', 'mid_level_action/simple_mid_ins_level_proposals_2021_1228_16h_17m_16s.txt'], store_dir = '/media/nvme1/linhanxi/data/ActionKnowledge/proposal_20211228/')
    elif mode=='t2':# 提取文本特征
        extract_proposal_fea_v2('/home/shiyuheng/ActionCLIP/mid_level_action/mid_level_act_proposalsV2_2023_1008_23h_59m_54s.txt', 'ViT-B/16', 
        store_dir='/home/shiyuheng/data/data_proposal_fea/', save_name='imagenet1k_1e-4')

    elif mode=='m':
        match_one(vid_pth='/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/official_val_mp4/val_256/kissing/OCQSpPkCAWw.pt', proposal_txt='')
    elif mode=='m2':
        # match_all('/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/batched_to_be_processed/*/*', '/media/nvme1/linhanxi/data/ActionKnowledge/proposal_20211219/combined.pt', 
        # {'old_root':'/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/batched_to_be_processed/', 'new_root':'/media/nvme1/linhanxi/data/k400/clip_match_results/batched_to_be_processed/'})
        # match_all('/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/batched_to_be_processed/*/*', '/media/nvme1/linhanxi/data/ActionKnowledge/part_ins_combined_1226.pt', 
        # {'old_root':'/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/batched_to_be_processed/', 'new_root':'/media/hdd/sda1/linhanxi/data/k400/clip_match_results/batched_to_be_processed_1227/'},
        # overwrite=True)
        match_all('/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/batched_to_be_processed/*/*', '/media/nvme1/linhanxi/data/ActionKnowledge/proposal_20211228/combined.pt', 
        {'old_root':'/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/batched_to_be_processed/', 'new_root':'/media/hdd/sda1/linhanxi/data/k400/clip_match_results/batched_to_be_processed_1228/'},
        overwrite=True)
    elif mode=='c':
        # combine(store_dir='/media/nvme1/linhanxi/data/ActionKnowledge/proposal_20211226/')
        combine(store_dir='/media/nvme1/linhanxi/data/ActionKnowledge/proposal_20211228/')
    elif mode=='probs2indices':
        probs2indices('/media/hdd/sda1/linhanxi/data/k400/clip_match_results/batched_to_be_processed_1228/train_frame/', '/media/hdd/sda1/linhanxi/data/k400/clip_match_results/topk_indices_20220110/train.pt', 50)
        probs2indices('/media/hdd/sda1/linhanxi/data/k400/clip_match_results/batched_to_be_processed_1228/val_frame/', '/media/hdd/sda1/linhanxi/data/k400/clip_match_results/topk_indices_20220110/val.pt', 50)
    elif mode=='new_match_4_new_proposals':
        '''
        new_match_4_new_proposals(
            'mid_level_action/mid_ins_level_proposals_handcraft_2022_0116_19h_49m_02s.txt',
            '/media/nvme1/linhanxi/data/ActionKnowledge/proposal_fea_cache/',
            '/media/hdd/sda1/linhanxi/data/k400/clip_match_results/batched_to_be_processed_cache/',
            '/media/nvme1/linhanxi/data/k400/fea_clip_vit_b_32/batched_to_be_processed/*/*',
            'mid_level_action/logs/ins_handcraft_tmpl.txt',
            setting={'num_segments':8, 'pth_to_save_dir':'/media/hdd/sda1/linhanxi/data/k400/clip_match_results/topk_indices_20220110/', 'nums_to_preserve':50}
            )
        '''
        new_match_4_new_proposals(
            'mid_level_action/hand_crafted_and_knowledge_base.txt', 
            '/media/sda1/shiyuheng/mid-level-action/cache/proposal_fea_cache/',
            '/media/sda1/shiyuheng/mid-level-action/cache/clip_match_results_batched_to_be_processed_cache/',
            '/media/sda1/shiyuheng/mid-level-action/finegym_Result_32f/fea_clip_RN50/batched_to_be_processed_visual/*/*',
            '/media/sda1/shiyuheng/mid-level-action/logs/mid_level_finegym_32f_match_probs_gen.txt',
            backbone='RN50',
            setting={'num_segments':32, 'unfolded_match_dir':'/media/sda1/shiyuheng/mid-level-action/finegym_match_L_hc_32f/'},
            store_mode='probs'
            )
    else:
        raise

    