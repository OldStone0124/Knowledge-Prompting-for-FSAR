import os
import torch
from torch import nn
import torch.nn.parallel

from opts import parse_opts
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalRandomCrop_TSNstyle
from datasets.kinetics_episode_w_knowledge import make_video_names, KineticsVideoList
from datasets.something_episode_w_knowledge import SomethingVideoList, make_something_video_names
from datasets.ucf101_episode_w_knowledge import UCFVideoList, make_ucf_video_names
from datasets.hmdb51_episode_w_knowledge import HMDBVideoList, make_hmdb_video_names
from datasets.diving48V2_episode_w_knowledge import diving48V2VideoList, make_diving48V2_video_names
from datasets.finegym_episode_w_knowledge import finegymVideoList, make_finegym_video_names

from utils import setup_logger, AverageMeter, count_acc, euclidean_metric
import time
from batch_sampler import CategoriesSampler
import torch.optim as optim
from clip_model import generate_model

from ipdb import set_trace
from utils import AverageMeter, calculate_accuracy, check_which_nan, check_tensor_nan

from lin_utils import backup_files
import copy
from torch.autograd import Variable
from lin_utils import join_multiple_txts
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class CLASSIFIERv2(CLASSIFIER):
    def __init__(self, input_dim, nclass):
        super().__init__(input_dim, nclass)
        self.temporal_model = nn.Sequential(
        nn.Conv1d(22348, 64, 1),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        )
    def forward(self, x):
        # set_trace()
        o = self.logic(self.fc(self.temporal_model(x.unsqueeze(-1)).squeeze(-1)))
        return o


def get_classifier_weights(embedding_shot, target_shot, lr=0.01, nepoch=5):
    classifier = CLASSIFIER(embedding_shot.size(1), opt.test_way)
    classifier.apply(weights_init)
    criterion = nn.NLLLoss()

    classifier.cuda()
    criterion.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.999))
    for i in range(nepoch):
        optimizer.zero_grad()
        output = classifier(embedding_shot)
        loss = criterion(output, target_shot)
        #print(loss.data)
        loss.backward()
        optimizer.step()

    return classifier.fc.weight.data


def train_epoch(support_data_loader, model, classifier, criterion, optimizer, opt, i):
    classifier.train()
    support_clip_embedding = torch.FloatTensor(opt.test_way*opt.shot*opt.n_samples_for_each_video, opt.emb_dim).cuda()
    support_clip_label = torch.LongTensor(opt.test_way*opt.shot*opt.n_samples_for_each_video).cuda()
    batch_size = opt.n_samples_for_each_video
    with torch.set_grad_enabled(opt.grad_enabled_in_embeddin), torch.cuda.amp.autocast(enabled=opt.is_amp):
    # with torch.no_grad(), torch.cuda.amp.autocast(enabled=opt.is_amp):
        cur_loc = 0
        for i,batch  in enumerate(support_data_loader):
            if opt.return_id:
                (data, label, clip_visfeas, iid) = batch
            else:
                (data, label, clip_visfeas) = batch
            # set_trace()
            batch_embedding = model(data.cuda(), clip_visfeas.cuda())
            cur_batch = batch_embedding.size(0)
            support_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            support_clip_label[cur_loc:cur_loc+cur_batch] = label.cuda()
            cur_loc += cur_batch
    # if opt.prototype_init and i==0:
    #     # support_clip_label.reshape(opt.shot,opt.test_way,opt.n_samples_for_each_video).transpose(0,1)# debug
    #     reshaped = support_clip_embedding.reshape(opt.shot,opt.test_way,opt.n_samples_for_each_video,-1).transpose(0,1).float()
    #     prototypes = torch.mean(reshaped, dim=(1,2))
    #     init_weight = nn.functional.normalize(prototypes,2,1) # L2 norm
    #     init_bias = torch.zeros(opt.test_way).float()
    #     classifier.fc.weight.data = init_weight
    #     classifier.fc.bias.data = init_bias
    # set_trace()

    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=opt.is_amp):
        output = classifier(support_clip_embedding)
        loss = criterion(output, support_clip_label)
    # loss.backward()
    # optimizer.step()
    opt.scaler.scale(loss).backward()
    opt.scaler.step(optimizer)
    opt.scaler.update()


def prototype_init(support_data_loader, model, classifier, opt):
    support_data_loader = copy.deepcopy(support_data_loader)
    support_clip_embedding = torch.FloatTensor(opt.test_way*opt.shot*opt.n_samples_for_each_video, opt.emb_dim).cuda()
    support_clip_label = torch.LongTensor(opt.test_way*opt.shot*opt.n_samples_for_each_video).cuda()
    batch_size = opt.n_samples_for_each_video
    with torch.set_grad_enabled(opt.grad_enabled_in_embeddin), torch.cuda.amp.autocast(enabled=opt.is_amp):
        cur_loc = 0
        for i, batch in enumerate(support_data_loader):
            if opt.return_id:
                (data, label, clip_visfeas, iid) = batch
            else:
                (data, label, clip_visfeas) = batch
            # set_trace()
            batch_embedding = model(data.cuda(), clip_visfeas.cuda())
            cur_batch = batch_embedding.size(0)
            support_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            support_clip_label[cur_loc:cur_loc+cur_batch] = label.cuda()
            cur_loc += cur_batch
    # support_clip_label.reshape(opt.shot,opt.test_way,opt.n_samples_for_each_video).transpose(0,1)# debug
    reshaped = support_clip_embedding.reshape(opt.shot,opt.test_way,opt.n_samples_for_each_video,-1).transpose(0,1).float()
    prototypes = torch.mean(reshaped, dim=(1,2))
    init_weight = nn.functional.normalize(prototypes,2,1) # L2 norm
    init_bias = torch.zeros(opt.test_way).float()
    classifier.fc.weight.data = init_weight
    classifier.fc.bias.data = init_bias
    # print('prototype init success!')


def val_epoch(query_data_loader, model, classifier, opt):
    classifier.eval()
    batch_size = opt.batch_size
    # `n_val_samples` is the number of sampled clips of a certain action video, 10 by default
    query_clip_embedding = torch.FloatTensor(opt.test_way * opt.query * opt.n_val_samples, opt.emb_dim).cuda()
    # query_clip_embedding.retain_grad()
    with torch.set_grad_enabled(opt.grad_enabled_in_embeddin), torch.cuda.amp.autocast(enabled=opt.is_amp):
        cur_loc = 0
        # set_trace()
        for i, batch in enumerate(query_data_loader):
            if opt.return_id:
                (data, label, clip_visfeas, iid) = batch
            else:
                (data, label, clip_visfeas) = batch
            # data.shape => torch.Size([64, 3, 16, 112, 112]) => [bs, C, clip_duration, H, W]
            batch_embedding = model(data.cuda(), clip_visfeas.cuda())
            # set_trace()
            # batch_embedding.shape => torch.Size([64, 512, 1, 1, 1])
            cur_batch = batch_embedding.size(0)
            query_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            # set_trace()
            cur_loc += cur_batch

        clip_logits = torch.exp(classifier(query_clip_embedding))
        #print(clip_logits)
        logits = clip_logits.reshape(opt.query * opt.test_way, opt.n_val_samples, -1).mean(dim=1)
        query_labels = torch.arange(opt.test_way).repeat(opt.query).cuda() # [0,1,...,opt.test_way-1] * opt.query
        # set_trace()
        acc, pred = count_acc(logits, query_labels)

    return acc


def interpreter_epoch(query_data_loader, model, classifier, opt, id2class_dict, proposals, logger):
    acc_meter = AverageMeter()
    proposals = np.array(proposals)
    datalist = query_data_loader.dataset.data
    labelInd2clsName_dic = {}
    for data in datalist:
        labelInd, clsName = data['label'].item(), id2class_dict[data['video_id']]
        labelInd2clsName_dic[labelInd] = clsName 
    clsNames = [labelInd2clsName_dic[labInd] for labInd in sorted(labelInd2clsName_dic.keys())]
    def require_module_grad(v):
        def backward_hook(module, grad_in, grad_out):
            v.grad_in = (grad_in[0].detach())
            v.grad_out = (grad_out[0].detach())
        v.register_backward_hook(backward_hook)
    classifier.eval()
    # `n_val_samples` is the number of sampled clips of a certain action video, 10 by default
    # query_clip_embedding = torch.FloatTensor(opt.test_way * opt.query * opt.n_val_samples, opt.emb_dim).cuda()
    # query_clip_embedding.retain_grad()
    query_labels = torch.arange(opt.test_way).repeat(opt.query).cuda() # [0,1,...,opt.test_way-1] * opt.query
    with torch.set_grad_enabled(opt.grad_enabled_in_embeddin), torch.cuda.amp.autocast(enabled=opt.is_amp):
        cur_loc = 0
        require_module_grad(model.module.scaled_cosine_simer)
        require_module_grad(model.module.knowledge_early_fuser)
        for i, batch in enumerate(query_data_loader):
            if opt.return_id:
                (data, label, clip_visfeas, iids) = batch
            else:
                (data, label, clip_visfeas) = batch
            # data.shape => torch.Size([64, 3, 16, 112, 112]) => [bs, C, clip_duration, H, W]
            clip_visfeas = Variable(clip_visfeas).cuda()
            clip_visfeas.requires_grad=True 
            batch_embedding, match, h = model.module.interpreter(data.cuda(), clip_visfeas, label)
            
            # batch_embedding.shape => torch.Size([64, 512, 1, 1, 1])
            cur_batch = batch_embedding.size(0)
            # query_clip_embedding[cur_loc:cur_loc+cur_batch] = batch_embedding.squeeze()
            cur_loc += cur_batch

            clip_logits = torch.exp(classifier(batch_embedding.squeeze()))
            # torch.autograd.grad(clip_logits.mean(), match, retain_graph=True)[0].max()
            for i,(this_logits, cur_labels, iid) in enumerate(zip(clip_logits, label, iids)):
                labelInd = cur_labels.item()
                this_top1predict = this_logits.argmax().item()
                judgement = this_top1predict==labelInd
                label_mask = torch.zeros_like(this_logits)
                label_mask[cur_labels] = 1 
                normalized_factore = 1/(this_logits*label_mask).mean().detach()
                retain_graph = i!=(clip_logits.size(0)-1)
                (this_logits*label_mask*normalized_factore).mean().backward(retain_graph=retain_graph)
                sortedIndices = sorted(range(len(this_logits)), key=lambda k: -this_logits[k])
                Pre_distribution = ['{}:{:.3}'.format(clsNames[ind], this_logits[ind]) for ind in sortedIndices]
                # warning: suppress time dimension
                topk_vals, topk_inds = match.grad_nonleaf[i].mean(-1).detach().topk(25)
                highlighted_proposals = proposals[topk_inds.tolist()]
                text_towrite = """
                vid:{vid}\t GT:{GT} 
                isright:{isright}\t Prediction:{Pre}\n
                Highlighted_proposals:{highlight}
                """.format(vid=iid, GT=labelInd2clsName_dic[labelInd], isright=judgement, Pre=' '.join(Pre_distribution), highlight=' '.join(highlighted_proposals),
                )
                logger.info(text_towrite)
                # match.grad_nonleaf.max()
                # match.grad_nonleaf.abs().sum()
                
            cur_acc, pred = count_acc(clip_logits.detach().cpu(), label)
            # cur_acc = 0
            acc_meter.update(cur_acc)
        h.remove()
        match.grad_nonleaf = None
    return acc_meter.avg


# **meta_test_episode**
@torch.no_grad()
def meta_test_episode(support_data_loader, query_data_loader, model, opt):
    acc_meter = AverageMeter()
    if opt.gradCAM or True:
        # prepare id2class_dict
        id2class_dict = {}
        if opt.dataset=='kinetics100':
            _list = ['data/kinetics100/data_splits/meta_test_filtered.txt', 'data/kinetics100/data_splits/meta_val.txt']
            for line in join_multiple_txts(_list):
                class_name, iid = line.split('/')[0], line.split('/')[1][:11]
                id2class_dict[iid] = class_name
            # prepare proposals
            proposal_list = ['/media/sda1/linhanxi/github/general-action-recog/mid_level_action/simple_mid_ins_level_proposals_k100_classes_2022_0317_13h_58m_58s.txt',
            '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/mid_level_act_proposalsV2_2022_0413_10h_29m_41s.txt']
        elif opt.dataset=='diving48V2':
            clsName_list_pth = '/home/linhanxi/diving48_240p/Diving48_vocab.json'
            import json
            loaded = json.load(open(clsName_list_pth,'rb'))
            clsName_list_pth = [" ".join(name) for name in loaded]
            _list = ['data/diving48V2/meta_test.txt', 'data/diving48V2/meta_val.txt']
            for i,line in enumerate(join_multiple_txts(_list)):
                class_name, iid = clsName_list_pth[int(line.split('/')[0])], line.split('/')[1]
                id2class_dict[iid] = class_name
            # prepare proposals
            proposal_list = [
                '/media/sda1/linhanxi/exp/NER/extract_proposals_V0.1_bugfix/extracted_instance_level_proposal',
                '/media/sda1/linhanxi/exp/NER/extract_proposals_V0.1_bugfix/extracted_part_level_proposal',
                '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/mid_level_act_proposalsV2_2022_0413_10h_29m_41s.txt'
            ]
            # set_trace()
        elif opt.dataset=='ucf101':
            _list = ['data/ucf101/data_splits/meta_test.txt', 'data/ucf101/data_splits/meta_val.txt']
            for line in join_multiple_txts(_list):
                class_name, iid = line.split('/')[0], line
                id2class_dict[iid] = class_name
            # prepare proposals
            proposal_list = ['/media/sda1/linhanxi/github/general-action-recog/mid_level_action/simple_mid_ins_level_proposals_k100_classes_2022_0317_13h_58m_58s.txt',
            '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/mid_level_act_proposalsV2_2022_0413_10h_29m_41s.txt']
        elif opt.dataset=='something':
            cates_names = [line.strip() for line in open('data/somethingv2/data_splits/test_classes.txt')]
            _list = ['data/somethingv2/data_splits/meta_test.txt']
            for i,line in enumerate(join_multiple_txts(_list)):
                class_name, iid = cates_names[int(line.strip().split()[-1])], line.strip().split()[0]
                id2class_dict[iid] = class_name
            # prepare proposals
            proposal_list = [
                '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/simple_mid_ins_level_proposals_ss_categories_2022_0304_10h_33m_13s.txt',
                '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/mid_level_act_proposalsV2_2022_0413_10h_29m_41s.txt'
            ]
        elif opt.dataset=='hmdb51':
            _list = ['data/hmdb_ARN/test.txt']
            for i,line in enumerate(join_multiple_txts(_list)):
                class_name, iid = " ".join(line.strip().split('/')[0].split('_')), line.strip()
                id2class_dict[iid] = class_name
            # prepare proposals
            # proposal_list = [
            #     '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/simple_mid_ins_level_proposals_ss_categories_2022_0304_10h_33m_13s.txt',
            #     '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/mid_level_act_proposalsV2_2022_0413_10h_29m_41s.txt'
            # ]
        elif opt.dataset=='finegym':
            _list = ['data/finegym/meta_test.txt']
            for line in join_multiple_txts(_list):
                class_name, iid = line.split('/')[0], line.split('/')[1]
                id2class_dict[iid] = class_name
                # if 'PlayingDaf/v_PlayingDaf_g01_c02' in line:
                #     set_trace()
            # prepare proposals
            proposal_list = [
                '/media/sda1/linhanxi/exp/NER/extract_proposals_V0.1_bugfix/extracted_instance_level_proposal',
                '/media/sda1/linhanxi/exp/NER/extract_proposals_V0.1_bugfix/extracted_part_level_proposal',
                '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/mid_level_act_proposalsV2_2022_0413_10h_29m_41s.txt'
            ]
        else:
            raise
        # set_trace()

        # prepare proposals
        # proposal_list = ['/media/sda1/linhanxi/github/general-action-recog/mid_level_action/simple_mid_ins_level_proposals_k100_classes_2022_0317_13h_58m_58s.txt',
        # '/media/sda1/linhanxi/github/general-action-recog/mid_level_action/mid_level_act_proposalsV2_2022_0413_10h_29m_41s.txt']
        # proposals = [line for line in join_multiple_txts(proposal_list)]
        # logger_gradCAM = setup_logger(
        # "gradCAM",
        # opt.result_path,
        # 0,
        # 'gradCAM.txt'
    # )

    # prepare lableInd2labelText dict
    lableInd2labelText_dict = {}
    for data in query_data_loader.dataset.data:
        # set_trace()
        lableInd2labelText_dict[data['label'].item()] = id2class_dict[data['video_id']]
    labelTexts = [lableInd2labelText_dict[i] for i in range(opt.test_way)]

    for i, batch in enumerate(query_data_loader):
        if opt.return_id:
            (data, label, clip_visfeas, iids) = batch
        else:
            (data, label, clip_visfeas) = batch
        text_tokens = clip.tokenize(["This is " + desc for desc in labelTexts]).cuda() 
        with torch.no_grad(),torch.cuda.amp.autocast():
            # torch.Size([64, 3, 16, 224, 224])
            bs, c, t, h, w = data.size()
            data = data.transpose(1,2).reshape(bs*t, c, h, w).cuda()
            # image_features = model.encode_image(data).reshape(bs,t,-1).mean(1).float()
            image_features = clip_visfeas.mean(1)
            # set_trace()
            text_features = model.encode_text(text_tokens).float()

        # cosine sim
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = model.logit_scale.detach().cpu().exp().numpy() * image_features.cpu().numpy() @ text_features.cpu().numpy().T
        # set_trace()
        similarity = np.exp(similarity)/sum(np.exp(similarity))
        cur_acc, pred = count_acc(torch.tensor(similarity), label)
        acc_meter.update(cur_acc)
        # set_trace()

    # acc = val_epoch(query_data_loader, model, classifier, opt)
    return acc_meter.avg


if __name__ == '__main__':
    opt = parse_opts()
    print(opt)

    '''amp'''
    opt.scaler = torch.cuda.amp.GradScaler(enabled=opt.is_amp)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.arch = '{}-{}'.format(opt.clip_model, opt.clip_model_depth)

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    else:
        raise ValueError(f'{opt.result_path} exists')

    '''backup'''
    if not os.path.exists(opt.this_launch_script):
        opt.this_launch_script = '/'.join(opt.this_launch_script.split('/')[1:])
    os.makedirs(os.path.join(opt.result_path, 'backup'))
    backup_files(
        [opt.this_launch_script,
        'datasets',
        'models',
        'building_model_w_knowledge.py',
        'tsl_fsv_w_knowledge.py',
        'clip_model.py',
        'opts.py',
        ],
        os.path.join(opt.result_path, 'backup')
        )
    # set_trace()
    
    # Setup logging system
    logger = setup_logger(
        "validation",
        opt.result_path,
        0,
        'results.txt'
    )
    logger.debug(opt)
    print(opt.lr)
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    torch.backends.cudnn.benchmark = True

    torch.manual_seed(opt.manual_seed)
    # **build model obj**
    import clip
    model, preprocess = clip.load("ViT-B/16")

    if opt.dataset == 'kinetics100':
        test_videos, test_labels = make_video_names(opt.test_video_path, opt.test_list_path)
    elif opt.dataset == 'something':
        test_videos, test_labels = make_something_video_names(opt.test_video_path, opt.test_list_path)
    elif opt.dataset == 'ucf101':
        test_videos, test_labels = make_ucf_video_names(opt.test_video_path, opt.test_list_path)
    elif opt.dataset == 'hmdb51':
        test_videos, test_labels = make_hmdb_video_names(opt.test_video_path, opt.test_list_path)
    elif opt.dataset == 'diving48V2':
        test_videos, test_labels = make_diving48V2_video_names(opt.test_video_path, opt.test_list_path)
    elif opt.dataset == 'finegym':
        test_videos, test_labels = make_finegym_video_names(opt.test_video_path, opt.test_list_path)

    # test_videos, test_labels is the the frm path and lab_ind `for each video`
    # set_trace()
    episode_sampler = CategoriesSampler(test_labels,
                                opt.nepisode, opt.test_way, opt.shot + opt.query)
    # default nepisode==500
    # episode_sampler samples the categories for testing(training) for each episode


    # set_trace()
    #print(model)

    norm_method = Normalize(preprocess.transforms[-1].mean, preprocess.transforms[-1].std)


    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])
    '''
    test_spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    '''
    test_spatial_transform = preprocess

    # test_temporal_transform = TemporalRandomCrop(opt.sample_duration)
    test_temporal_transform = eval('''{'dense':TemporalRandomCrop, 'sparse':TemporalRandomCrop_TSNstyle}''')[opt.sample_mode](opt.sample_duration)


    episode_time = AverageMeter()
    accuracies = AverageMeter()

    for i, batch_idx in enumerate(episode_sampler):
        #print(batch_idx)
        k = opt.test_way * opt.shot
        support_videos = [test_videos[j] for j in batch_idx[:k]]
        support_labels = torch.arange(opt.test_way).repeat(opt.shot)

        query_videos = [test_videos[j] for j in batch_idx[k:]]
        query_labels = torch.arange(opt.test_way).repeat(opt.query)

        if opt.dataset == 'kinetics100':
            support_data_loader = torch.utils.data.DataLoader(
                KineticsVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_data_loader = torch.utils.data.DataLoader(
                KineticsVideoList(
                    query_videos,
                    query_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

        elif opt.dataset == 'something':
            support_data_loader = torch.utils.data.DataLoader(
                SomethingVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_data_loader = torch.utils.data.DataLoader(
                SomethingVideoList(
                    query_videos,
                    query_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

        elif opt.dataset == 'ucf101':
            support_data_loader = torch.utils.data.DataLoader(
                UCFVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_data_loader = torch.utils.data.DataLoader(
                UCFVideoList(
                    query_videos,
                    query_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

        elif opt.dataset == 'hmdb51':
            support_data_loader = torch.utils.data.DataLoader(
                HMDBVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_data_loader = torch.utils.data.DataLoader(
                HMDBVideoList(
                    query_videos,
                    query_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.dataset == 'diving48V2':
            support_data_loader = torch.utils.data.DataLoader(
                diving48V2VideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_data_loader = torch.utils.data.DataLoader(
                diving48V2VideoList(
                    query_videos,
                    query_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.dataset == 'finegym':
            support_data_loader = torch.utils.data.DataLoader(
               finegymVideoList(
                    support_videos,
                    support_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_samples_for_each_video,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            query_data_loader = torch.utils.data.DataLoader(
                finegymVideoList(
                    query_videos,
                    query_labels,
                    spatial_transform=test_spatial_transform,
                    temporal_transform=test_temporal_transform,
                    n_samples_for_each_video=opt.n_val_samples,
                    args=opt),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        
                

        # set_trace()
        end_time = time.time()
        acc = meta_test_episode(support_data_loader, query_data_loader, model, opt)
        accuracies.update(acc)
        episode_time.update(time.time() - end_time)

        logger.info('Episode: {0}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  i + 1,
                  batch_time=episode_time,
                  acc=accuracies))