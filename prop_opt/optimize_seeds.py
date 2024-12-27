import torch
import numpy as np
import pandas as pd
import os
import argparse
from prop_opt.utils import smiles_to_latent_vec, latent_to_smiles, get_similarity, get_mol_properties
from prop_opt.optimizers import random_walk, random_ray, gradient_ascent, langevin_dynamics
from prop_pred.train_multi_task import get_model
from moflow.mflow.models.hyperparams import Hyperparameters
from moflow.mflow.utils.model_utils import load_model
from moflow.data.transform_zinc250k import zinc250_atomic_num_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_atoms', type=int, default=38, help='Number of max atoms in the molecule')
    parser.add_argument('--model_dir', type=str, help='Path to model directory')
    parser.add_argument("--snapshot-path", "-snapshot", type=str)
    parser.add_argument("--hyperparams-path", type=str, default='moflow-params.json')

    parser.add_argument('--model_type', type=str, help='Type of model to use')
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument('--depth', type=int, help='Depth of the model')
    parser.add_argument('--output_dim', type=int, help='Output dimension')
    parser.add_argument('--p', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on')
    parser.add_argument('--save_path', type=str, default='.')
    
    parser.add_argument('--optimizer', type=str, default='random_walk', help='Optimizer to use')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of optimization steps')
    parser.add_argument('--alpha', type=float, default=0.1, help='Step size for optimization')
    parser.add_argument('--beta', type=float, default=0.1, help='Noise scale for Langevin dynamics')
    parser.add_argument('--target_idx', type=int, default=0, help='Index of the target property')
    parser.add_argument('--seed_path', type=str, help='Path to seed smiles csv')
    parser.add_argument('--output_path', type=str, help='Path to save optimized smiles')
    
    args = parser.parse_args()

    print('Get model')
    prop_pred = get_model(args.model_type, args.input_dim, args.depth, args.output_dim, args.p)
    prop_pred.load_state_dict(torch.load(os.path.join(args.save_path, 'model.pt')))
    prop_pred.to(args.device)

    print('Get MoFlow model')
    ## Load Pre-trained MoFlow model
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    print("loading hyperparamaters from {}".format(hyperparams_path))
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)
    model = model.to('cuda')

    print('Load seed smiles')
    seed_df = pd.read_csv(args.seed_path)
    seed_smiles = seed_df['smiles']

    print('Optimize seeds')
    NEW_SMILES = []; SEED_SMILES = []; SIMILARITY = []; LOGP = []; QED = []; SA = []; STEPS = []
    for seed_smile in seed_smiles:
        initial_z = smiles_to_latent_vec([seed_smile], model, device='cuda', max_atoms=args.max_atoms)
        if args.optimizer == 'random_walk':
            new_latents = random_walk(initial_z, args.num_steps, alpha=args.alpha)
        elif args.optimizer == 'random_ray':
            new_latents = random_ray(initial_z, args.num_steps, seed_smile, alpha=args.alpha)
        elif args.optimizer == 'gradient_ascent':
            new_latents = gradient_ascent(initial_z, args.num_steps, prop_pred, args.target_idx, alpha=args.alpha)
        elif args.optimizer == 'langevin_dynamics':
            new_latents = langevin_dynamics(initial_z, args.num_steps, prop_pred, args.target_idx, alpha=args.alpha, beta=args.beta)
        else:
            raise ValueError('Invalid optimizer')
        
        for i, new_latent in enumerate(new_latents):
            new_smile = latent_to_smiles(new_latent, model, zinc250_atomic_num_list, device='cuda')
            if new_smile is None:
                continue
            similarity = get_similarity(seed_smile, new_smile)
            logp, qed, sa = get_mol_properties(new_smile)
            NEW_SMILES.append(new_smile)
            SEED_SMILES.append(seed_smile)
            SIMILARITY.append(similarity)
            LOGP.append(logp)
            QED.append(qed)
            SA.append(sa)
            STEPS.append(i)
    
    results = pd.DataFrame({'seed_smiles': SEED_SMILES, 'new_smiles': NEW_SMILES, 'similarity': SIMILARITY, 'logp': LOGP, 'qed': QED, 'sa': SA, 'steps': STEPS})
    results.to_csv(args.output_path, index=False)