cd few-shot-video-classification/ && CUDA_VISIBLE_DEVICES=1 python tsl_fsv_w_knowledge.py --test_video_path  /media/sda1/linhanxi/data/ucf101/UCF101_rawimage/ \
--manual_seed 10 \
--test_list_path data/ucf101/data_splits/meta_test.txt \
--dataset ucf101 \
--train_crop random \
--n_samples_for_each_video 10 \
--knowledge_model dwconv_fc \
--n_val_samples 10 \
--clip_model r2plus1d_w_knowledge \
--clip_model_depth 34 \
--n_threads 4 \
--result_path /media/sda1/linhanxi/exp/few-shot-vid-cls/test_ucf/knowledge_model/dwconv_fc/v0.3_5w5s_sparseSample_32frms_sparseBugfix_PoolBugfix_regBugfix_metaTest_PartV2.1_testCLIPvisBugfix_selfatten_learningPropV0.1bugfix2_secDrop005_ksize3_ablationLinear_save32 \
--shot 5 \
--test_way 5 \
--query 5 \
--resume_path /media/sda1/linhanxi/exp/few-shot-vid-cls/finetune_ucfCMN/knowledge_model/dwconv_fc/v0.3_clsBugFix_embedOrig_ablation_origBugfix_sparseSample_32frms_sparseBugfix_PoolBugfix_selfatten_secDrop005_ksize3_gymPartDivingPart_ablationLinear/save_32.pth \
--emb_dim 491 \
--batch_size 64 \
--lr 0.01 \
--nepoch 10 \
--CLIP_visual_fea_reg "/home/linhanxi/home_data/clip_related/ucf_CMN/batched_to_be_processed_visual/fea_clip_ViT-B-16_unfolded_32frms/*/*" \
--proposals_fea_pth /media/sda1/linhanxi/data/CLIP_related/action_knowledge/ucfARN/proposal_fea_PartV2_gymPart_divingPart_cache.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 32 --n_finetune_classes 64 --is_w_knowledge \
--is_amp --this_launch_script $0 --ablation_removeOrig --print_freq 200 --sample_mode sparse --temporal_modeling TSM2 --ablation_onlyLinear #--grad_enabled_in_embeddin #-ablation_onlyCLIPvisfea #--CLIP_visual_fea_preload
# --emb_dim 448 \
# --emb_dim 477 \
# --emb_dim 450 \
# --emb_dim 514 \