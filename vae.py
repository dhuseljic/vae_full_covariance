"""Naive implementation of VAE with a full covariance Gaussian for the approx. dist

[1] Kingma, D. P., & Welling, M. (2019). An Introduction to Variational Autoencoders. Foundations and Trends®
    in Machine Learning, 12(4), 307–392. https://doi.org/10.1561/2200000056
"""
import math
import tqdm
import pylab as plt

import torch
import torch.nn as nn

import torchvision
from utils.datasets import load_mnist
from utils.vae import Unflatten

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_vae():
    train_ds, val_ds, test_ds = load_mnist()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

    encoder = Encoder()
    decoder = nn.Sequential(
        nn.Linear(392, 784), nn.ReLU(),
        nn.Linear(784, 784), nn.Sigmoid(), Unflatten(1, 28, 28)
    )

    optimizer_encoder = torch.optim.Adam(encoder.parameters())
    optimizer_decoder = torch.optim.Adam(decoder.parameters())

    encoder.to(device)
    decoder.to(device)

    fixed_noise = torch.randn(32, 392, device=device)

    # Algorithm from [1]
    for X_batch, _ in tqdm.tqdm(train_loader):
        X_batch = X_batch.to(device)

        mean, logvar, lvals = encoder(X_batch)
        eps, z = encoder.reparam(mean, logvar, lvals)

        log_qz = -torch.sum(
            .5*(eps**2 + math.log(2 * math.pi) + torch.sqrt(torch.exp(logvar))), -1)
        log_pz = -torch.sum(.5*(z**2 + math.log(2*math.pi)), -1)

        p = decoder(z)
        log_px = torch.sum(X_batch * torch.log(p) + (1 - X_batch) * torch.log(1 - p), dim=(1, 2, 3))

        loss = -(torch.mean(log_px) + torch.mean(log_pz) - torch.mean(log_qz))
        loss.backward()
        optimizer_decoder.step()
        optimizer_decoder.zero_grad()
        optimizer_encoder.step()
        optimizer_encoder.zero_grad()

    with torch.no_grad():
        img = torchvision.utils.make_grid(decoder(fixed_noise))
        plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
        plt.show()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_latent = 392
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 784)
        self.l2_mean = nn.Linear(784, self.n_latent)  # mean of q(z|x)
        self.l2_logvar = nn.Linear(784, self.n_latent)  # logvar of q(z|x)

        # Determine the number of outputs for the triangular matrix
        self.tri_idx_row, self.tri_idx_col = torch.tril_indices(row=self.n_latent, col=self.n_latent)
        idx = (self.tri_idx_row != self.tri_idx_col)  # Delete diagonal
        self.tri_idx_row = self.tri_idx_row[idx]
        self.tri_idx_col = self.tri_idx_col[idx]

        self.l2_Lvals = nn.Linear(784, len(self.tri_idx_col))  # cholesky of cov

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)

        mean = self.l2_mean(x)
        logvar = self.l2_logvar(x)
        Lvals = self.l2_Lvals(x)

        return mean, logvar, Lvals

    def reparam(self, mean, logvar, L_vals):
        device = mean.device
        batch_size = mean.size(0)

        # Transform Lvals into a triangular matrix
        L = torch.zeros(batch_size, self.n_latent, self.n_latent, device=device)
        L[:, self.tri_idx_row, self.tri_idx_col] = L_vals
        L += torch.diag_embed(torch.exp(logvar))  # Diagonal matrix for batch

        # sample noise from mutlivariate Gaussian
        eps = torch.randn((batch_size, self.n_latent), device=device)

        # Use batchwise matrix multiplication
        sample = mean + torch.bmm(L, eps.unsqueeze(-1)).squeeze()

        return eps, sample


if __name__ == "__main__":
    train_vae()
