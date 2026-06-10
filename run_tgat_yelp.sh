#!/usr/bin/env bash
set -euo pipefail

python train_tgb1_fair.py \
  --model_kind tgat \
  --dataset Yelp-TPA \
  --ns_q 1000 \
  --epochs 30 \
  --evaluate-every -1 \
  --batch_size 512 \
  --eval_batch_size 64 \
  --eval_candidate_batch_size 8192 \
  --node_feat_dim 64 \
  --time_feat_dim 64 \
  --rel_dim 64 \
  --predictor_hidden_dim 128 \
  --num_neighbors 20 \
  --num_layers 2 \
  --num_heads 2 \
  --lr 0.0005 \
  --gpu 0 \
  > tgat_tpa.log 2>&1 &
wait

python train_tgb1_fair.py \
  --model_kind tgat \
  --dataset Yelp-NOLA \
  --ns_q 1000 \
  --epochs 30 \
  --evaluate-every -1 \
  --batch_size 512 \
  --eval_batch_size 64 \
  --eval_candidate_batch_size 8192 \
  --node_feat_dim 64 \
  --time_feat_dim 64 \
  --rel_dim 64 \
  --predictor_hidden_dim 128 \
  --num_neighbors 20 \
  --num_layers 2 \
  --num_heads 2 \
  --lr 0.0005 \
  --gpu 0 \
  > tgat_nola.log 2>&1 &
wait

python train_tgb1_fair.py \
  --model_kind tgat \
  --dataset Yelp-PHL \
  --ns_q 1000 \
  --epochs 30 \
  --evaluate-every -1 \
  --batch_size 512 \
  --eval_batch_size 64 \
  --eval_candidate_batch_size 8192 \
  --node_feat_dim 64 \
  --time_feat_dim 64 \
  --rel_dim 64 \
  --predictor_hidden_dim 128 \
  --num_neighbors 20 \
  --num_layers 2 \
  --num_heads 2 \
  --lr 0.0005 \
  --gpu 0 \
  > tgat_phl.log 2>&1 &
wait
