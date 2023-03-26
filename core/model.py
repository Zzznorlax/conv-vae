import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, d_size: int = 32, latent_dim: int = 256, input_size: int = 28):
        super(Encoder, self).__init__()

        self.d_size = d_size

        self.conv1 = nn.Conv2d(in_ch, d_size, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_size, d_size * 2**1, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(d_size * 2**1, d_size * 2**2, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(d_size * 2**2 * 4 * 4, 256)

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.d_size * 2**2 * 4 * 4)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 256, d_size: int = 32):
        super(Decoder, self).__init__()

        self.d_size = d_size

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, d_size * 2**2 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(d_size * 2**2, d_size * 2**1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(d_size * 2**1, d_size * 2**0, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(d_size * 2**0, 1, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, z: torch.Tensor):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, self.d_size * 2**2, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))

        # input image is normalized to [0, 1], so the output of the decoder should also be normalized to [0, 1].
        x = torch.sigmoid(self.deconv3(x))
        return x


class ConvVAE(nn.Module):
    def __init__(self, in_ch: int = 3, d_size: int = 32, latent_dim: int = 256, input_size: int = 28):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(in_ch, d_size, latent_dim, input_size)
        self.decoder = Decoder(latent_dim, d_size)

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


def deconv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    h_w = np.array(h_w)
    kernel_size = np.array((kernel_size, kernel_size))
    stride = np.array((stride, stride))
    pad = np.array((pad, pad))
    dil = np.array((dilation, dilation))

    out_shape = (h_w - 1) * stride + dil * (kernel_size - 1) + 2 * pad
    return tuple(out_shape)
