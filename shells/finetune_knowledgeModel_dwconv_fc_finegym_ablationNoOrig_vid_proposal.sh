cd few-shot-video-classification/ && CUDA_VISIBLE_DEVICES=1 python finetune_metatrain_w_knowledge.py --video_path /media/sda1/shiyuheng/data/gym/subaction_frames/ \
--train_list_path data/finegym/meta_train2.txt \
--val_video_path /media/sda1/shiyuheng/data/gym/subaction_frames/ \
--val_list_path data/finegym/meta_trainval.txt \
--dataset finegym \
--n_classes 400 \
--n_finetune_classes 99 \
--model r2plus1d_w_knowledge \
--knowledge_model dwconv_fc \
--model_depth 34 \
--batch_size 32 \
--n_threads 4 \
--checkpoint 1 \
--val_every 1 \
--train_crop random \
--n_samples_for_each_video 8 \
--n_val_samples 10 \
--weight_decay 0.001 \
--layer_lr 0.001 0.001 0.001 0.001 0.001 0.1 \
--ft_begin_index 0 \
--result_path /media/sda1/shiyuheng/actionKnowledgeXfewshot/few-shot-video-classification/result/train_video_proposal \
--CLIP_visual_fea_reg "/home/shiyuheng/data/data_CLIP_visual_fea/finegym/batched_to_be_processed_visual/fea_clip_ViT-B-16_unfolded_32frms/*" \
--proposals_fea_pth /home/shiyuheng/data/data_proposal_fea/video_proposal.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum  32 --is_w_knowledge --is_amp --dropout_w_knowledge 0.9 \
--ablation_removeOrig --this_launch_script $0 --print_freq 200 --sample_mode sparse --n_epochs 100 --temporal_modeling TSM1 --sample_duration 16 #--ablation_onlyCLIPvisfea #--embeddin_bias #--ablation_onlyCLIPvisfea #--l1regu --l1_factor 0.00005 #--CLIP_visual_fea_preload #--with_clip_zeroshot --way_to_use_zeroshot naive_sum
 #--CLIP_visual_fea_preload # 
 #9--pretrain_path /home/linhanxi/nfs_mount/30902_hdd_sda1_linhanxi/pretrained_weights/few-shot-video-classification/fsv_pretrained_model/r25d34_sports1m.pth \