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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    transform = torchvision.transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST('~/.datasets', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

    vae = VariationalAutoencoder()
    optimizer = torch.optim.Adam(vae.parameters())

    vae.to(device)
    fixed_noise = torch.randn(32, 392, device=device)

    # Algorithm from [1]
    for X_batch, _ in tqdm.tqdm(train_loader):
        X_batch = X_batch.to(device)

        mean, logvar, lvals = vae.encode(X_batch)
        eps, z = vae.reparam(mean, logvar, lvals)

        log_qz = -torch.sum(.5*(eps**2 + math.log(2 * math.pi) + logvar), -1)
        log_pz = -torch.sum(.5*(z**2 + math.log(2*math.pi)), -1)

        p = vae.decoder(z)
        log_px = torch.sum(X_batch * torch.log(p) + (1 - X_batch) * torch.log(1 - p), dim=(1, 2, 3))

        loss = -(torch.mean(log_px) + torch.mean(log_pz) - torch.mean(log_qz))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        img = torchvision.utils.make_grid(vae.decoder(fixed_noise))
        plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
        plt.show()


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_latent = 392

        # Encoder Example
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

        self.decoder = nn.Sequential(
            nn.Linear(392, 784), nn.ReLU(),
            nn.Linear(784, 784), nn.Sigmoid(), Unflatten(1, 28, 28)
        )

    def encode(self, x):
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

class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


if __name__ == "__main__":
    main()
