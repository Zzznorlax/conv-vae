import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, d_size: int = 32, latent_dim: int = 256, input_size: int = 28):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, d_size, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_size, d_size * 2, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z: torch.Tensor):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))

        # input image is normalized to [0, 1], so the output of the decoder should also be normalized to [0, 1].
        x = torch.sigmoid(self.deconv2(x))
        return x


class ConvVAE(nn.Module):
    def __init__(self, in_ch: int = 3, d_size: int = 32, latent_dim: int = 256, input_size: int = 28):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(in_ch, d_size, latent_dim, input_size)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
