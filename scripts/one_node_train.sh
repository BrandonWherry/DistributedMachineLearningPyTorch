#! /bin/bash
train_time=${1:-0.25}
model_name=${2:-model}
b_sz=${3:-64}


# one node
torchrun                        \
--standalone                    \
--nproc_per_node=gpu            \
multi_node_trainer.py           \
--train_time="$train_time"      \
--model_name="$model_name.pt"   \
--batch_size="$b_sz"            \
2>&1 | tee "training_saves/$model_name.log"