# __Distributed Machine Learning in PyTorch__
This project is intended to explore PyTorch's Distributed ML capabilities, specifically their __Distributed Data Parallel__ strategy (DDP).


## One Worker Script
`bash scripts/one_node_train.sh TRAIN_TIME  SAVE_NAME  BATCH_SIZE`

__Example__: (run from within this dir)

`bash scripts/one_node_train.sh 3.0 single_node 64`

## Multi Worker Script
`bash scripts/one_node_train.sh TRAIN_TIME WORKER_NUM WORLD_SIZE SAVE_NAME BATCH_SIZE`

__Example__: (run from within this dir)

`bash scripts/multi_node_train.sh 3.0 0 4 multi_node 64`
`bash scripts/multi_node_train.sh 3.0 1 4 multi_node 64`
`bash scripts/multi_node_train.sh 3.0 2 4 multi_node 64`
`bash scripts/multi_node_train.sh 3.0 3 4 multi_node 64`

The above code would be run on 4 workers, for 3 hours, and the checkpoints would be named "multi_node" (as specificed by SAVE_NAME arg) with BATCH_SIZE of 64.

See useful_server_commands.txt for more examples.

__Args:__

`TRAIN_TIME = Total training time in hours`  
`WORKER_NUM = worker number`  
`WORLD_SIZE = Total worker count`    
`SAVE_NAME = name for saving checkpoints, metrics`  
`BATCH_SIZE = Batch Size per device `

# __Testing Results & Conclusions for Project__


For PyTorch DDP testing, I used 1/50 of imageNet on a modified VGG19 model, training on 1, 2, 4, and 8 AWS EC2 instances.

Instance type = g4dn.2xlarge - 8 vCPUs - 1 Nvidia T4 GPU

In each experiment, I trained for 3 hours.