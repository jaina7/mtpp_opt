import torch
import numpy as np

def random_walk(z, steps, alpha=0.1):
    new_latents = []
    cur_latent = z

    for i in range(steps):
        new_latent = cur_latent + alpha*torch.randn_like(cur_latent)
        new_latents.append(new_latent.detach().cpu())
        cur_latent = new_latent
    new_latents = torch.stack(new_latents)

    return new_latents

def random_ray(z, steps, seed_smiles, alpha=0.1):
    new_latents = []
    noise = torch.randn_like(z)
    cur_latent = z

    for i in range(steps):
        new_latent = cur_latent + alpha*noise
        new_latents.append(new_latent.detach().cpu())
        cur_latent = new_latent
    new_latents = torch.stack(new_latents)

    return new_latents

def gradient_ascent(z, steps, prop_pred, idx, alpha=0.1):
    new_latents = []
    cur_latent = z

    for i in range(steps):
        cur_latent.requires_grad = True
        preds = prop_pred(cur_latent.squeeze(0))[:, idx]
        grad = torch.autograd.grad(preds, cur_latent)[0]
        new_latent = cur_latent + alpha*grad
        new_latents.append(new_latent.detach().cpu())
        cur_latent = new_latent
    new_latents = torch.stack(new_latents)

    return new_latents

def langevin_dynamics(z, steps, model, idx, alpha=0.1, beta=0.1):
    new_latents = []
    cur_latent = z

    for i in range(steps):
        cur_latent.requires_grad = True
        preds = model(cur_latent.squeeze(0))[:, idx]
        grad = torch.autograd.grad(preds, cur_latent)[0]
        noise = torch.randn_like(cur_latent)
        new_latent = cur_latent + alpha*grad + beta*noise
        new_latents.append(new_latent.detach().cpu())
        cur_latent = new_latent
    new_latents = torch.stack(new_latents)

    return new_latents