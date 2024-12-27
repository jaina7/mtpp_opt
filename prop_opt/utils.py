from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed #, MolLogP
from rdkit.Chem import rdMolDescriptors
from moflow.mflow.utils.sascorer import calculateScore

import os, sys
import torch
from moflow.mflow.models.model import MoFlow, rescale_adj
from moflow.mflow.models.utils import construct_mol
from moflow.data.smile_to_graph import GGNNPreprocessor

## Covert smiles to latent vectors
@torch.no_grad()
def smiles_to_latent_vec(smiles, model, device='cuda', max_atoms=38):
    nodes, adjs = [], []
    smiles_processor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        cannonical_smile, mol = smiles_processor.prepare_smiles_and_mol(mol)
        node, adj = smiles_processor.get_input_features(mol)
        nodes.append(node)
        adjs.append(adj)
    nodes = torch.tensor(nodes).to(device)
    adjs = torch.tensor(adjs).to(device)

    bs = nodes.shape[0]
    adj_normalized = rescale_adj(adjs)
    zs, _ = model(adjs, nodes, adj_normalized)
    z = torch.cat([zs[0].view(bs, -1), zs[1].view(bs, -1)], dim=1)

    return z

@torch.no_grad()
def latent_to_smiles(z, model, atomic_num_list, device='cuda'):
    adj, x = model.reverse(z)
    adj = adj.squeeze(0)
    x = x.squeeze(0)
    adj = adj.cpu().numpy()
    x = x.cpu().numpy()
    mol = construct_mol(adj, x, atomic_num_list)
    smi = Chem.MolToSmiles(mol)
    return Chem.MolFromSmiles(smi)

def get_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_mol_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    logp = Chem.Descriptors.MolLogP(mol)
    qed_score = qed(mol)
    sa_score = calculateScore(mol)
    return logp, qed_score, sa_score