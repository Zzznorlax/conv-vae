import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.mean()


def elbo_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float = 1) -> torch.Tensor:
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')

    kld_loss = kl_divergence(mu, logvar)

    print(kld_loss.item() * kld_weight, recon_loss.item())
    elbo = recon_loss + kld_loss * kld_weight
    return elbo
