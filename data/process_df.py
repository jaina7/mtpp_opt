from rdkit import Chem
import os, sys
sys.path.append('..')
sys.path.append('../moflow')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time

from moflow.data.smile_to_graph import GGNNPreprocessor
from moflow.data.transform_zinc250k import transform_fn_zinc250k, get_val_ids

from moflow.mflow.models.hyperparams import Hyperparameters
from moflow.mflow.utils.model_utils import load_model, get_latent_vec
from moflow.mflow.models.model import MoFlow, rescale_adj
from prop_preprocessor import PreProcess

IDENTITIY_IDX = [0]
POSITIVE_IDX = [1]
NEGATIVE_IDX = [2]
EXPONENTIAL_IDX = [3]
BOUNDED_IDX = [4]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to csv file containing smiles and properties')
    parser.add_argument('--max_atoms', type=int, help='Number of max atoms in the molecule')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument("--snapshot-path", "-snapshot", type=str)
    parser.add_argument("--hyperparams-path", type=str, default='moflow-params.json')
    parser.add_argument("--save_path", type=str, default='.')

    args = parser.parse_args()

    ## Load Pre-trained MoFlow model
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    print("loading hyperparamaters from {}".format(hyperparams_path))
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)
    model = model.to('cuda')

    ## Load molecules and pre-prcoess properties
    df = pd.read_csv(args.path)
    smiles = df['smiles']
    smiles_processor = GGNNPreprocessor(out_size=args.max_atoms, kekulize=True)
    nodes, adjs = [], []
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        cannonical_smile, mol = smiles_processor.prepare_smiles_and_mol(mol)
        node, adj = smiles_processor.get_input_features(mol)
        nodes.append(node)
        adjs.append(adj)
    nodes = np.array(nodes)
    adjs = np.array(adjs)

    valid_ids = get_val_ids()
    train_ids = [i for i in range(len(nodes)) if i not in valid_ids]

    train_nodes = nodes[train_ids]; train_adjs = adjs[train_ids]
    valid_nodes = nodes[valid_ids]; valid_adjs = adjs[valid_ids]

    df = df.drop(columns=['smiles'])
    properties = df.values
    prop_preprocessor = PreProcess(IDENTITIY_IDX, POSITIVE_IDX, NEGATIVE_IDX, EXPONENTIAL_IDX, BOUNDED_IDX)
    train_props = prop_preprocessor(properties[train_ids])
    valid_props = prop_preprocessor(properties[valid_ids])

    ## Convert molecules to MoFlow latent embeddings
    train_nodes, train_adjs, train_props = transform_fn_zinc250k((train_nodes, train_adjs, train_props))
    valid_nodes, valid_adjs, valid_props = transform_fn_zinc250k((valid_nodes, valid_adjs, valid_props))

    train_nodes, train_adjs, train_props = torch.from_numpy(train_nodes), torch.from_numpy(train_adjs), torch.from_numpy(train_props)
    valid_nodes, valid_adjs, valid_props = torch.from_numpy(valid_nodes), torch.from_numpy(valid_adjs), torch.from_numpy(valid_props)

    train_graph_dataset = torch.utils.data.TensorDataset(train_nodes, train_adjs)
    valid_graph_dataset = torch.utils.data.TensorDataset(valid_nodes, valid_adjs)
    train_graph_loader = torch.utils.data.DataLoader(train_graph_dataset, batch_size=args.batch, shuffle=True)
    valid_graph_loader = torch.utils.data.DataLoader(valid_graph_dataset, batch_size=args.batch, shuffle=False)
    
    train_latents = []
    with torch.no_grad():
        for batch in train_graph_loader:
            nodes, adjs = batch
            adj_normalized = rescale_adj(adjs)
            nodes, adjs, adj_normalized = nodes.to('cuda'), adjs.to('cuda'), adj_normalized.to('cuda')
            bs = nodes.shape[0]

            zs, _ = model(adjs, nodes, adj_normalized)
            z = torch.cat([zs[0].view(bs, -1), zs[1].view(bs, -1)], dim=1)
            train_latents.append(z.clone().detach())

    valid_latents = []
    with torch.no_grad():
        for batch in valid_graph_loader:
            nodes, adjs = batch
            adj_normalized = rescale_adj(adjs)
            nodes, adjs, adj_normalized = nodes.to('cuda'), adjs.to('cuda'), adj_normalized.to('cuda')
            bs = nodes.shape[0]

            zs, _ = model(adjs, nodes, adj_normalized)
            z = torch.cat([zs[0].view(bs, -1), zs[1].view(bs, -1)], dim=1)
            valid_latents.append(z.clone().detach())

    train_latents = torch.cat(train_latents, dim=0)
    valid_latents = torch.cat(valid_latents, dim=0)

    ## Save the latent embeddings, properties and preprocessing objects
    torch.save(train_latents, os.path.join(args.save_path, 'train_latents.pt'))
    torch.save(valid_latents, os.path.join(args.save_path, 'test_latents.pt'))
    torch.save(train_props, os.path.join(args.save_path, 'train_props.pt'))
    torch.save(valid_props, os.path.join(args.save_path, 'test_props.pt'))
    torch.save(prop_preprocessor, os.path.join(args.save_path, 'prop_preprocessor.pth'))