# __Distributed Machine Learning in PyTorch__
This project is intended to explore PyTorch's Distributed ML capabilities, specifically their __Distributed Data Parallel__ strategy (DDP)

## Single Worker Script
`bash scripts/single_node_train.sh TRAIN_TIME  SAVE_NAME  BATCH_SIZE`

__Example__: (run from within this dir)

`bash scripts/single_node_train.sh 3.0 single_node 64`

## Multi Worker Script
`bash scripts/single_node_train.sh TRAIN_TIME WORKER_NUM WORLD_SIZE SAVE_NAME BATCH_SIZE`

__Example__: (run from within this dir)

`bash scripts/multi_node_train.sh 3.0 0 4 multi_node 64`
`bash scripts/multi_node_train.sh 3.0 1 4 multi_node 64`
`bash scripts/multi_node_train.sh 3.0 2 4 multi_node 64`
`bash scripts/multi_node_train.sh 3.0 3 4 multi_node 64`

The above code would be run on 4 workers, for 3 hours, and the checkpoints would be named "multi_node" (as specificed by SAVE_NAME arg) with BATCH_SIZE of 64.

__Args:__

`TRAIN_TIME = Total training time in hours`  
`WORKER_NUM = worker number`  
`WORLD_SIZE = Total worker count`    
`SAVE_NAME = name for saving checkpoints, metrics`  
`BATCH_SIZE = Batch Size per device `

# __Results & Conclusions for Project__
