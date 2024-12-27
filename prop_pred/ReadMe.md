## Model Types
1. mlp: Network with fully connected layers
2. resnet: Network with Residual blocks of user-defined width
3. wrn: Network with Residual blocks of constant width

## Multi-target training
```
python train_multi_task.py --model_type mlp --input_dim 6156 --depth 3 --output_dim 166 --p 0.1 --batch_size 128 --device cuda --data_path ./data/ --num_epochs 100 --save_path ./saved/mlp --log_file log.txt --seed 0
```
model_type: mlp, resnet, wrn
input_dim: Number of input features
depth: Number of layers or residual blocks in the network
output_dim: Number of output targets
p: Dropout probability
batch_size: Batch size
device: cuda or cpu
data_path: Path to the data folder
num_epochs: Number of epochs
save_path: Path to save the model
log_file: Path to save the log file for comparison
seed: Random seed

## Single-target training
```
python train_single_task.py --model_type mlp --input_dim 6156 --depth 3 --p 0.1 --batch_size 128 --device cuda --data_path ./data/ --num_epochs 100 --save_path ./saved/mlp --log_file log.txt --seed 0 --prop_id 0
```
model_type: mlp, resnet, wrn
input_dim: Number of input features
depth: Number of layers or residual blocks in the network
p: Dropout probability
batch_size: Batch size
device: cuda or cpu
data_path: Path to the data folder
num_epochs: Number of epochs
save_path: Path to save the model
log_file: Path to save the log file for comparison
seed: Random seed
prop_id: Property ID to train the model