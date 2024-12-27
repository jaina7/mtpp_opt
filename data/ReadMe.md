## Optimize seed molecules
Store the list of molecules in a csv file. The csv file must contain molecular smiles in a column `smiles`. 
Pre-trained MoFlow model and hyperparameters are required to generate latent embeddings and properties.
Pre-trained property prediction models are required to predict properties on the latent embeddings for gradient-based optimization.

## Optimizers
1. Random walk: Randomly sample molecules from the latent space and evaluate properties.
2. Random ray: Randomly sample a direction in the latent space and evaluate properties.
3. Gradient ascent: Use property prediction gradients to optimize properties.
4. Langevin dynamics: Use property prediction gradients and noise to optimize properties.

## Usage
```
python optimize_seeds.py --model_dir <Path to MoFlow model directory> --snapshot-path <Path to MoFlow snapshot> --hyperparams-path <Path to MoFlow hyperparam file> --model_type <Type of Prop Pred model> --input_dim <Input dimension> --depth <Depth of the model> --output_dim <Output dimension> --p <Dropout probability> --device <Device model is trained on> --save_path <Path to pre-trained property prediction models> --optimizer <Optimizer to use> --num_steps <Number of optimization steps> --alpha <Step size for optimization> --beta <Noise scale for Langevin dynamics> --target_idx <Index of the target property> --seed_path <Path to seed smiles csv> --output_path <Path to save optimized smiles>
```