#! /bin/bash
train_time=${1:-0.25}
model_name=${2:-model}
b_sz=${3:-64}


# Executing torchrun on a single worker
torchrun                        \
--standalone                    \
--nproc_per_node=1            \
multi_node_trainer.py           \
--train_time="$train_time"      \
--model_name="$model_name.pt"   \
--batch_size="$b_sz"            \
2>&1 | tee "training_saves/$model_name.log"


