import torch
import torch.utils.data
from torch import nn, optim
from model import VAE
from train import VAETrainer
from torch.nn import functional as F

def loss_function(recon_x, x, mu, logvar):
        # Reconstruction error
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # Kulback-Leibler divergence between the returned distribution and a standard Gaussian
        KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        return BCE + KLD

def train(cfg):
    model = VAE(z_dim=cfg.Z_DIM)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    evaluator = VAETrainer(
        cfg,
        model=model,
        optimizer=optimizer,
        criterion=loss_function
    )
    evaluator.train()

def test(cfg, input_tensor):
    model = VAE(z_dim=cfg.Z_DIM)
    optimizer = optim.Adam(model.parameter(), lr=cfg.LEARNING_RATE)
    criterion = loss_function
    