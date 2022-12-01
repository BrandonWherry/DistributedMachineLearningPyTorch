#! /bin/bash
train_time=${1:-0.25}
worker_num=${2:-0}
world_size=${3:-1}
model_name=${4:-model}
batch_size=${5:-64}

torchrun                            \ 
--nproc_per_node=gpu                \ 
--nnodes="$world_size"              \ 
--node_rank="$worker_num"           \ 
--rdzv_id=123                       \
--rdzv_backend=c10d                 \
--rdzv_endpoint=172.31.12.214:29500 \
multi_node_trainer.py               \
--train_time="$train_time"          \ 
--model_name="$model_name.pt"       \
--batch_size="$batch_size"          \
2>&1 | tee "training_saves/$model_name.log"