cd few-shot-video-classification/ && CUDA_VISIBLE_DEVICES=0 python finetune_metatrain_w_knowledge.py --video_path /home/linhanxi/home_data/k400_CMNsplit/ \
--train_list_path data/kinetics100/data_splits/meta_train_filtered.txt \
--val_video_path /home/linhanxi/home_data/k400_CMNsplit/ \
--val_list_path data/kinetics100/data_splits/trainclasses_val_filtered.list \
--dataset kinetics \
--n_classes 400 \
--n_finetune_classes 64 \
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
--result_path /media/sda1/shiyuheng/actionKnowledgeXfewshot/few-shot-video-classification/results/kinetics/train_knowbase_1e-4_finegym_ins \
--CLIP_visual_fea_reg "/media/nvme1n1p1/linhanxi/data/clip_related/k400CMN/fea_clip_ViT-B-16_unfolded_32frms/*" \
--proposals_fea_pth /home/shiyuheng/data/data_proposal_fea/knowbase_1e-4_finegym_ins.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 32 --is_w_knowledge --is_amp --dropout_w_knowledge 0.9 \
--ablation_removeOrig --this_launch_script $0 --print_freq 200 --sample_mode sparse --n_epochs 50 --temporal_modeling TSM1  #--with_clip_zeroshot --way_to_use_zeroshot adaptive_fuseV2
 #--CLIP_visual_fea_preload 
 #--result_path /media/sda1/linhanxi/exp/few-shot-vid-cls/finetune_kinetics/knowledge_model/dwconv_fc/v0.3_clsBugFix_embedOrig_dropout0.9_ablation_origBugfix_sparseSample_TSM1_32frms_sparseBugfix_PoolBugfix_PartProposaV2.1_DEBUG_selfatten_ksize3_gymPartDivingPart \
 #--pretrain_path /home/linhanxi/nfs_mount/30902_hdd_sda1_linhanxi/pretrained_weights/few-shot-video-classification/fsv_pretrained_model/r25d34_sports1m.pth \