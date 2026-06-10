#!/usr/bin/env bash
set -euo pipefail

python train_tgb1_fair.py \
  --model_kind dygformer \
  --dataset Yelp-TPA \
  --ns_q 1000 \
  --epochs 30 \
  --evaluate-every -1 \
  --batch_size 512 \
  --eval_batch_size 32 \
  --eval_candidate_batch_size 4096 \
  --node_feat_dim 64 \
  --time_feat_dim 64 \
  --rel_dim 64 \
  --predictor_hidden_dim 128 \
  --channel_embedding_dim 32 \
  --patch_size 2 \
  --max_input_sequence_length 64 \
  --num_layers 2 \
  --num_heads 2 \
  --lr 0.0005 \
  --gpu 0 \
  > dygformer_tpa.log 2>&1 &
wait

python train_tgb1_fair.py \
  --model_kind dygformer \
  --dataset Yelp-NOLA \
  --ns_q 1000 \
  --epochs 30 \
  --evaluate-every -1 \
  --batch_size 512 \
  --eval_batch_size 32 \
  --eval_candidate_batch_size 4096 \
  --node_feat_dim 64 \
  --time_feat_dim 64 \
  --rel_dim 64 \
  --predictor_hidden_dim 128 \
  --channel_embedding_dim 32 \
  --patch_size 2 \
  --max_input_sequence_length 64 \
  --num_layers 2 \
  --num_heads 2 \
  --lr 0.0005 \
  --gpu 0 \
  > dygformer_nola.log 2>&1 &
wait

python train_tgb1_fair.py \
  --model_kind dygformer \
  --dataset Yelp-PHL \
  --ns_q 1000 \
  --epochs 30 \
  --evaluate-every -1 \
  --batch_size 512 \
  --eval_batch_size 32 \
  --eval_candidate_batch_size 4096 \
  --node_feat_dim 64 \
  --time_feat_dim 64 \
  --rel_dim 64 \
  --predictor_hidden_dim 128 \
  --channel_embedding_dim 32 \
  --patch_size 2 \
  --max_input_sequence_length 64 \
  --num_layers 2 \
  --num_heads 2 \
  --lr 0.0005 \
  --gpu 0 \
  > dygformer_phl.log 2>&1 &
wait
