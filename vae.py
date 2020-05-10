import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import pylab as plt


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.flatten()),
    ])
    train_ds = torchvision.datasets.MNIST('~/.datasets', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, num_workers=4, pin_memory=True)

    # Train
    vae = VAE()
    optimizer = torch.optim.Adam(vae.parameters())
    vae.to(device)

    running_loss = 0
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)

        optimizer.zero_grad()
        recon, mean, logvar, z = vae(X_batch)

        loss_reconstruction = X_batch * torch.log(recon) + (1 - X_batch) * torch.log(1 - recon)
        loss_reconstruction = torch.mean(- torch.sum(loss_reconstruction, dim=-1))
        loss_kl = torch.mean(-0.5 * torch.sum((1 + logvar - mean**2 - logvar.exp()), -1))

        loss = loss_reconstruction + loss_kl
        loss.backward()
        optimizer.step()
        running_loss += loss * X_batch.size(0)

    recon, _, _, _ = vae(X_batch)
    plt.imshow(recon[0].detach().view(28, 28))
    plt.show()


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.encoder_mean = nn.Linear(64, 2)
        self.encoder_logvar = nn.Linear(64, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar

    def reparam(self, mean, logvar):
        eps = torch.randn(mean.shape, device=mean.device)
        return (0.5 * logvar.exp()) * eps + mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparam(mean, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar, z


if __name__ == "__main__":
    main()
