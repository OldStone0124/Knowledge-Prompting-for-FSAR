cd few-shot-video-classification/ && CUDA_VISIBLE_DEVICES=1 python tsl_fsv_w_knowledge.py --test_video_path  /home/linhanxi/diving48_240p/frames_240p/ \
--manual_seed 10 \
--test_list_path data/diving48V2/meta_test.txt \
--dataset diving48V2 \
--train_crop random \
--n_samples_for_each_video 10 \
--knowledge_model dwconv_fc \
--n_val_samples 10 \
--clip_model r2plus1d_w_knowledge \
--clip_model_depth 34 \
--n_threads 4 \
--result_path /media/sda1/linhanxi/exp/few-shot-vid-cls/test_diving48V2/knowledge_model/dwconv_fc/v0.3_5w5s_sparseSample_32frms_sparseBugfix_PoolBugfix_regBugfix_metaTest_PartV2.1_testCLIPvisBugfix_selfatten_learningPropV0.1bugfix2_secDrop005_ksize3_clipvisfeaBase_save31 \
--shot 5 \
--test_way 5 \
--query 5 \
--resume_path /media/sda1/linhanxi/exp/few-shot-vid-cls/finetune_diving48V2/knowledge_model/dwconv_fc/v0.3_clsBugFix_embedOrig_ablation_origBugfix_sparseSample_32frms_sparseBugfix_PoolBugfix_selfatten_secDrop005_ksize3_gymPartDivingPart_clipvisfeaBase/save_31.pth \
--emb_dim 491 \
--batch_size 64 \
--lr 0.01 \
--nepoch 10 \
--CLIP_visual_fea_reg "/media/nvme1n1p1/linhanxi/data/clip_related/diving48V2/batched_to_be_processed_visual/fea_clip_ViT-B-16_unfolded_32frms/*" \
--proposals_fea_pth /media/sda1/linhanxi/data/CLIP_related/action_knowledge/KineticsCMN/proposal_fea_PartV2_gymPart_divingPart_cache.pt  \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 32 --n_finetune_classes 36 --is_w_knowledge \
--is_amp --this_launch_script $0 --ablation_removeOrig --print_freq 200 --sample_mode sparse --temporal_modeling TSM2 --ablation_onlyCLIPvisfea #--clip_duration 32 #--grad_enabled_in_embeddin #--CLIP_visual_fea_preload
# --emb_dim 448 \
# --emb_dim 477 \
# --emb_dim 450 \
# --emb_dim 514 \