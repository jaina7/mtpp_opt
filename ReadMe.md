# Multi-target property prediction and optimization

## Instal MoFlow and Libs
Follow the instructions to download MoFlow generative model from {[here](https://github.com/calvin-zcx/moflow)}. 
1. Install the pre-requisite libraries for MoFlow
2. Download the pre-trained model and specify the path in `moflow` and `data` scripts
 

## Data preparation
To generate latent embeddings and pre-processed properties, first download pretrained Moflow model.
```
cd data
python process_df.py --path <Path to csv file> --max_atoms 38 --batch_size 128 --snapshot-path <Path to folder of pre-trained MoFlow> --hyperparams-path <Path to hyperparam file of Moflow> --save_path <Path to save the latent embeddings and properties as torch tensors.>
```
The csv file must contain molecular smiles in a column `smiles`. The other columns should be the molecular properties that will be predicted. Global variables that specify which category each property needs to be specified in `data/process_df.py`. The data is saved after pre-processing in train and test set splits.

After processing, data loaders can by calling `get_train_loader` and `get_test_loader` in `data/get_data_loaders.py`. The data has the shape,
```
train_loader = get_train_loader()
batch = next(iter(train_loader))
latents, props = batch
latents # M x 6156
props # M x P
```

## Train Property Prediction models
To train the property prediction models, follow the instructions in `prop_pred/ReadMe.md`. The models can be trained on the latent embeddings and properties generated in the previous step. The models can be trained in single-target or multi-target mode. The models can be trained with different architectures such as mlp, resnet, wrn. The models can be trained with different hyperparameters such as depth, dropout probability, batch size, number of epochs, and random seed.

After training, the models can be used to predict properties on the test set. The predictions can be evaluated using metrics such as MSE and correlation. The script stores loss and correlation values in a log file. We compare different models along with their hyperparameters using the log file.

Log File header is
```
{model_type}_{input_dim}_{depth}_{output_dim}_{p}, Train Loss, Val Loss, Test Loss, Train Corr, Val Corr, Test Corr
```

## Optimization
Pre-trained MoFlow model and hyperparameters are required to generate latent embeddings and properties. Pre-trained property prediction models are required to predict properties on the latent embeddings for gradient-based optimization.
Create a csv file with the list of molecules to optimize. The csv file must contain molecular smiles in a column `smiles`. For example, sort the molecules by a property and select the top molecules to optimize. The csv file should contain the worst molecules to optimize.
Optimized molecules are stored in a csv file with discovered molecules and their properties.