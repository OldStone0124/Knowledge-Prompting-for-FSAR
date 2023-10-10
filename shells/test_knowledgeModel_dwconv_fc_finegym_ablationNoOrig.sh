cd few-shot-video-classification/ && CUDA_VISIBLE_DEVICES=1 python tsl_fsv_w_knowledge.py --test_video_path  /media/sda1/shiyuheng/data/gym/subaction_frames/ \
--manual_seed 10 \
--test_list_path data/finegym/meta_test.txt \
--dataset finegym \
--train_crop random \
--n_samples_for_each_video 10 \
--knowledge_model dwconv_fc \
--n_val_samples 10 \
--clip_model r2plus1d_w_knowledge \
--clip_model_depth 34 \
--n_threads 4 \
--result_path /media/sda1/shiyuheng/actionKnowledgeXfewshot/few-shot-video-classification/result/test_knowbase_6e-4/ \
--shot 5 \
--test_way 5 \
--query 5 \
--resume_path /media/sda1/shiyuheng/actionKnowledgeXfewshot/few-shot-video-classification/result/knowbase_6e-4/save_32.pth \
--emb_dim 491 \
--batch_size 64 \
--lr 0.01 \
--nepoch 10 \
--CLIP_visual_fea_reg "/home/shiyuheng/data/data_CLIP_visual_fea/finegym/batched_to_be_processed_visual/fea_clip_ViT-B-16_unfolded_32frms/*" \
--proposals_fea_pth /home/shiyuheng/data/data_proposal_fea/knowbase_6e-4.pt \
--CLIP_visual_arch "ViT-B/16"  --clip_visfea_sampleNum 32 --n_finetune_classes 99 --is_w_knowledge \
--ablation_removeOrig --is_amp --this_launch_script $0 --print_freq 200 --sample_mode sparse --temporal_modeling TSM2 --clip_duration 16 #--embeddin_bias #--grad_enabled_in_embeddin --return_id --gradCAM --grad_enabled_in_match #--grad_enabled_in_embeddin #-ablation_onlyCLIPvisfea #--CLIP_visual_fea_preload
# --emb_dim 448 \
# --emb_dim 477 \
# --emb_dim 450 \
# --emb_dim 514 \
#--ablation_onlyCLIPvisfea
#/media/sda1/linhanxi/data/CLIP_related/action_knowledge/KineticsCMN/proposal_fea_PartV2_gymPart_divingPart_cache.pt  \