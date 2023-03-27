import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from core.config import Settings, get_settings
from core.model import ConvVAE
from core.dataset import DatasetBase
from core.utils.model import get_latest_ckpt
from core.utils.log import get_logger


def generate_samples(model, img_size: int = 32):
    with torch.no_grad():
        z = torch.randn(100, model.latent_dim)
        samples = model.decoder(z).cpu().numpy()

    # Rescale pixel values from [0, 1] to [0, 255]
    samples = (samples * 255).astype(np.uint8)

    # Reshape samples to a 10 by 10 grid of images
    samples = samples.reshape(10, 10, img_size, img_size)
    samples = np.transpose(samples, (0, 2, 1, 3))
    samples = samples.reshape(10 * img_size, 10 * img_size)

    # Display the grid of images
    plt.figure(figsize=(8, 8))
    plt.imshow(samples, cmap='gray')
    plt.axis('off')
    plt.show()


def eval_vae(opt: Settings, grid_size: int = 10):

    logger = get_logger()

    val_transforms = None

    val_dataset = DatasetBase(opt.VALIDATION_DATASET_PATH, transform=val_transforms, suffix=[opt.EXT], img_size=opt.IMG_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=grid_size**2, shuffle=True)

    model = ConvVAE(in_ch=1, input_size=opt.IMG_SIZE, latent_dim=opt.LATENT_SIZE, d_size=opt.D_SIZE)

    # loads state from previous checkpoint
    ckpt = get_latest_ckpt(opt.CKPT_DIR)
    if ckpt:
        state_dict = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state'])

    model.eval()

    samples = None

    for _, img in enumerate(val_loader):

        img = img.to(opt.DEVICE)
        recon_imgs, _, _ = model(img)
        samples = recon_imgs.detach().numpy()
        # samples = img.detach().numpy()

        break

    if samples is not None:
        # Rescale pixel values from [0, 1] to [0, 255]
        samples = (samples * 255).astype(np.uint8)

        # Reshape samples to a 10 by 10 grid of images
        samples = samples.reshape(grid_size, grid_size, opt.IMG_SIZE, opt.IMG_SIZE)
        samples = np.transpose(samples, (0, 2, 1, 3))
        samples = samples.reshape(grid_size * opt.IMG_SIZE, grid_size * opt.IMG_SIZE)

        # Display the grid of images
        plt.figure(figsize=(8, 8))
        plt.imshow(samples, cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    opt = get_settings()

    eval_vae(opt)
