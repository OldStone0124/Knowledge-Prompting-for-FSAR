cd few-shot-video-classification/ && CUDA_VISIBLE_DEVICES=0 python tsl_fsv_w_knowledge.py --test_video_path  /home/linhanxi/home_data/k400_CMNsplit_30fps/ \
--manual_seed 10 \
--test_list_path data/kinetics100/data_splits/meta_test_filtered.txt \
--dataset kinetics100 \
--train_crop random \
--n_samples_for_each_video 10 \
--n_val_samples 10 \
--clip_model r2plus1d_w_knowledge \
--knowledge_model dwconv_fc \
--clip_model_depth 34 \
--n_threads 4 \
--result_path /media/sda1/shiyuheng/actionKnowledgeXfewshot/few-shot-video-classification/results/kinetics/test_knowbase_1e-4_finegym_ins \
--shot 5 \
--test_way 5 \
--query 5 \
--resume_path /media/sda1/shiyuheng/actionKnowledgeXfewshot/few-shot-video-classification/results/kinetics/train_knowbase_1e-4_finegym_ins/save_7.pth \
--emb_dim 448 \
--batch_size 64 \
--lr 0.01 \
--nepoch 10 \
--CLIP_visual_fea_reg "/media/nvme1n1p1/linhanxi/data/clip_related/k400CMN/fea_clip_ViT-B-16_unfolded_32frms/*" \
--proposals_fea_pth /home/shiyuheng/data/data_proposal_fea/knowbase_1e-4_finegym_ins.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 32 --n_finetune_classes 64 --is_w_knowledge \
--is_amp --this_launch_script $0 --ablation_removeOrig --print_freq 200 --sample_mode sparse --temporal_modeling TSM2 #--grad_enabled_in_embeddin --return_id --gradCAM --grad_enabled_in_match #--tune_specific #--grad_enabled_in_embeddin #-ablation_onlyCLIPvisfea #--CLIP_visual_fea_preload
#--resume_path /media/sda1/linhanxi/exp/few-shot-vid-cls/finetune_kinetics/knowledge_model/dwconv_fc/v0.3_clsBugFix_embedOrig_dropout0.9_ablation_origBugfix_sparseSample_TSM1_32frms_sparseBugfix_PoolBugfix_PartProposaV2.1_DEBUG_selfatten_ksize3_gymPartDivingPart/save_22.pth \
#--result_path /media/sda1/linhanxi/exp/few-shot-vid-cls/test_kinetics/knowledge_model/dwconv_fc/v0.3_drop0.9_5w5s_sparseSample_TSM2_32frms__sparseBugfix_PoolBugfix_regBugfix_metaTest_PartV2.1_testCLIPvisBugfix_selfatten_ksize3_moreEp_gymPartDivingPart_save22 \