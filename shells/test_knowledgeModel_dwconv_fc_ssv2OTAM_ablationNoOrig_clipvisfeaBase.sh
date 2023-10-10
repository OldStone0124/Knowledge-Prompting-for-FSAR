cd few-shot-video-classification/ && CUDA_VISIBLE_DEVICES=1 python tsl_fsv_w_knowledge.py --test_video_path  /home/linhanxi/nfs_mount/30902_hdd_sda1_linhanxi/data/ss_v2/extracted_frms_OTAM/ssv2_256x256q5_extraVal/ \
--manual_seed 10 \
--test_list_path data/somethingv2/data_splits/meta_test.txt \
--dataset something \
--train_crop random \
--n_samples_for_each_video 10 \
--n_val_samples 10 \
--clip_model r2plus1d_w_knowledge \
--knowledge_model dwconv_fc \
--clip_model_depth 34 \
--n_threads 4 \
--result_path /media/sda1/linhanxi/exp/few-shot-vid-cls/test_ssv2OTAM/knowledge_model/dwconv_fc/v0.3_drop0.9_5w5s_sparseSample_TSM2_32frms__sparseBugfix_PoolBugfix_regBugfix_metaTest_PartV2.1_testCLIPvisBugfix_selfatten_ksize3_gymPartDivingPart_clipvisfeaBase_save31 \
--shot 5 \
--test_way 5 \
--query 5 \
--resume_path /media/sda1/linhanxi/exp/few-shot-vid-cls/finetune_ssv2OTAM/knowledge_model/dwconv_fc/v0.3_clsBugFix_embedOrig_dropout0.9_ablation_origBugfix_sparseSample_TSM1_32frms_sparseBugfix_PoolBugfix_PartProposaV2.1_DEBUG_selfatten_ksize3_gymPartDivingPart_clipvisfeaBase/save_31.pth \
--emb_dim 512 \
--batch_size 64 \
--lr 0.01 \
--nepoch 10 \
--CLIP_visual_fea_reg "/home/linhanxi/home_data/clip_related/ssv2_OTAM/batched_to_be_processed_visual/fea_clip_ViT-B-16_unfolded_32frms/*" \
--proposals_fea_pth /media/sda1/linhanxi/data/CLIP_related/action_knowledge/ssv2OTAM/proposal_fea_new_cache.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 32 --n_finetune_classes 64 --is_w_knowledge \
--is_amp --this_launch_script $0 --ablation_removeOrig --print_freq 200 --sample_mode sparse --temporal_modeling TSM2 --ablation_onlyCLIPvisfea   #--grad_enabled_in_embeddin #-ablation_onlyCLIPvisfea #--CLIP_visual_fea_preload