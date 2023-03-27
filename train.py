import os
import random
import torch
import wandb
import math
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from core.config import Settings, get_settings
from core.model import ConvVAE
from core.dataset import DatasetBase, dataset_statistic
from core.metric import elbo_loss, kl_divergence
from core.utils.model import get_latest_ckpt
from core.utils.log import get_logger


def train_vae(opt: Settings):

    logger = get_logger()

    os.makedirs(opt.CKPT_DIR, exist_ok=True)

    aug_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=opt.AUG_ROTATION),
        transforms.RandomAutocontrast(p=opt.AUG_AUTO_CONTRAST),
        transforms.ColorJitter(brightness=opt.AUG_COLOR_JITTER, contrast=opt.AUG_COLOR_JITTER, saturation=opt.AUG_COLOR_JITTER),
    ])  # type: ignore

    train_dataset = DatasetBase(opt.DATASET_PATH, transform=aug_transforms, suffix=[opt.EXT], size=opt.DATASET_SIZE, img_size=opt.IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=opt.BATCH_SIZE, shuffle=True)

    val_transforms = None

    val_dataset = DatasetBase(opt.VALIDATION_DATASET_PATH, transform=val_transforms, suffix=[opt.EXT], size=opt.DATASET_SIZE, img_size=opt.IMG_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=opt.BATCH_SIZE, shuffle=False)

    model = ConvVAE(in_ch=1, input_size=opt.IMG_SIZE, latent_dim=opt.LATENT_SIZE, d_size=opt.D_SIZE)

    optimizer = Adam(model.parameters(), lr=opt.LR)

    # custom loss function
    def _loss_func(x, y, mu, logvar):

        return elbo_loss(x, y, mu, logvar, kld_weight=opt.KLD_WEIGHT)

    # loads state from previous checkpoint
    if opt.RESUME:
        ckpt = get_latest_ckpt(opt.CKPT_DIR)
        if ckpt:
            state_dict = torch.load(ckpt)
            model.load_state_dict(state_dict['model_state'])
            cur_itrs = state_dict['cur_itrs']
            best_loss = state_dict['best_loss']
            optimizer.load_state_dict(state_dict['optimizer_state'])
            logger.info("resumed from {}, iteration {}, best loss: {}".format(state_dict, cur_itrs + 1, best_loss))

    device = opt.DEVICE
    model.to(device)

    wandb.watch(model, log="all", log_freq=100)

    best_loss = 1e10
    cur_itrs = 0
    cum_batch = 0
    train_total_loss = 0
    train_total_kld = 0
    for epoch in range(opt.EPOCHS):

        for batch_idx, (img, label) in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()

            img = img.to(device)

            recon_img, mu, logvar = model(img)

            loss = _loss_func(recon_img, img, mu, logvar)
            train_total_loss += loss.item()
            train_total_kld += kl_divergence(mu, logvar).item()

            loss.backward()
            optimizer.step()

            # logger.info("epoch ({}/{}), batch {}, Training Loss: {}".format(epoch + 1, opt.EPOCHS, batch_idx + 1, loss.item()))

            # saves checkpoint
            if (batch_idx + 1) % opt.CKPT_INTERVAL == 0:

                ckpt_types = ["latest"]
                if best_loss > loss:
                    best_loss = loss
                    ckpt_types.append("best")

                state = {
                    "cur_itrs": cur_itrs,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": best_loss,
                }

                for ckpt_type in ckpt_types:
                    ckpt_path = os.path.join(opt.CKPT_DIR, "ckpt_{}.pth".format(ckpt_type))
                    torch.save(state, ckpt_path)
                    wandb.save(ckpt_path)

            cum_batch += 1
            cur_itrs += img.shape[0]

            # performs validation and logs to wandb
            if (batch_idx + 1) % opt.LOG_INTERVAL == 0:

                val_cum_batch = 0
                val_total_loss = 0
                val_total_kld = 0
                model.eval()
                with torch.no_grad():
                    for _, (val_img, _) in enumerate(val_loader):

                        val_img = val_img.to(device)

                        val_recon_img, val_mu, val_logvar = model(val_img)

                        val_loss = _loss_func(val_recon_img, val_img, val_mu, val_logvar)
                        val_total_loss += val_loss.item()
                        val_total_kld += kl_divergence(val_mu, val_logvar).item()

                        val_cum_batch += 1

                val_avg_kld = val_total_kld / val_cum_batch
                train_avg_kld = train_total_kld / cum_batch

                val_avg_loss = val_total_loss / val_cum_batch
                train_avg_loss = train_total_loss / cum_batch

                logger.info("epoch ({}/{}), iteration {}, Validation Loss: {}".format(epoch + 1, opt.EPOCHS, cur_itrs, val_avg_loss))
                logger.info("epoch ({}/{}), iteration {}, Training Loss: {}".format(epoch + 1, opt.EPOCHS, cur_itrs, train_avg_loss))

                wandb.log({
                    "val_kld": val_avg_kld,
                    "val_loss": val_avg_loss,
                    "train_kld": train_avg_kld,
                    "train_loss": train_avg_loss,
                    "iteration": cur_itrs,
                })
                train_total_loss = 0
                train_total_kld = 0
                cum_batch = 0

        # generates samples and logs image to wandb
        sample_loader = DataLoader(val_dataset, batch_size=opt.SAMPLE_GRID_SIZE**2)

        samples = None
        input_imgs = None
        with torch.no_grad():

            model.eval()

            for _, (img, _) in enumerate(sample_loader):

                img = img.to(opt.DEVICE)
                recon_imgs, _, _ = model(img)
                samples = recon_imgs.cpu().detach().numpy()
                input_imgs = img.cpu().detach().numpy()

                break

            if samples is not None:
                # Rescale pixel values from [0, 1] to [0, 255]
                samples = (samples * 255).astype(np.uint8)

                # Reshape samples to a grid of images
                samples = samples.reshape(opt.SAMPLE_GRID_SIZE, opt.SAMPLE_GRID_SIZE, opt.IMG_SIZE, opt.IMG_SIZE)
                samples = np.transpose(samples, (0, 2, 1, 3))
                samples = samples.reshape(opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE, opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE)

            if input_imgs is not None:
                # Rescale pixel values from [0, 1] to [0, 255]
                input_imgs = (input_imgs * 255).astype(np.uint8)

                # Reshape input images to a grid of images
                input_imgs = input_imgs.reshape(opt.SAMPLE_GRID_SIZE, opt.SAMPLE_GRID_SIZE, opt.IMG_SIZE, opt.IMG_SIZE)
                input_imgs = np.transpose(input_imgs, (0, 2, 1, 3))
                input_imgs = input_imgs.reshape(opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE, opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE)

        # plots latent space
        latent_space_sample_loader = DataLoader(val_dataset, batch_size=opt.SAMPLE_LATENT_NUM)

        z = None
        labels = None
        fig = None
        with torch.no_grad():

            model.eval()

            for _, (img, label) in enumerate(latent_space_sample_loader):

                img = img.to(opt.DEVICE)

                sample_mu, sample_logvar = model.encoder(img)
                z = model.reparameterize(sample_mu, sample_logvar).cpu().numpy()

                labels = np.array(label).astype(int)

                break

            if z is not None and labels is not None:
                # Use sklearn.manifold.TSNE to reduce the dimensionality of the latent vectors to 2D
                tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                z_tsne = tsne.fit_transform(z)

                # Plot the 2D latent vectors using a scatter plot
                fig, ax = plt.subplots()
                scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap='tab10')
                legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
                ax.add_artist(legend)
                plt.colorbar(scatter)

        # plots latent distribution
        eps_sample_loader = DataLoader(val_dataset, batch_size=opt.SAMPLE_EPS_IMG_NUM)

        recon_list = []
        recon_imgs = None
        with torch.no_grad():

            model.eval()

            for _, (imgs, _) in enumerate(eps_sample_loader):

                imgs = imgs.to(opt.DEVICE)

                for img in imgs:
                    sample_mu, sample_logvar = model.encoder(img.unsqueeze(0))

                    # repeats the mean and logvar to create a one-hot encoding matrix of size (latent_size, latent_size)
                    sample_mu, sample_logvar = sample_mu.repeat(opt.LATENT_SIZE, 1), sample_logvar.repeat(opt.LATENT_SIZE, 1)

                    # Fill the diagonal elements with ones to create a one-hot encoding
                    eps = torch.eye(opt.LATENT_SIZE).unsqueeze(0).to(opt.DEVICE)

                    z = model.sample_z(sample_mu, sample_logvar, eps)

                    output_img_grid = model.decoder(z).cpu().detach().numpy()  # shape: (latent_size, 1, img_size, img_size)

                    num_rc = int(np.sqrt(opt.LATENT_SIZE))
                    output_img_grid = output_img_grid.reshape(num_rc, num_rc, 1, opt.IMG_SIZE, opt.IMG_SIZE)
                    output_img_grid = output_img_grid.transpose(0, 3, 1, 4, 2)
                    output_img_grid = output_img_grid.reshape(1, num_rc * opt.IMG_SIZE, num_rc * opt.IMG_SIZE)

                    recon_list.append(output_img_grid)

                break

        wandb.log({
            "epoch": epoch + 1,
            "samples": wandb.Image(samples, caption="Samples"),
            "inputs": wandb.Image(input_imgs, caption="Inputs"),
            "tsne_plot": wandb.Image(fig),
            "z_samples": wandb.Image(np.array(recon_list), caption="Z Samples"),
        })

    logger.info("Training finished")


def dataset_stats(opt: Settings):

    train_dataset = DatasetBase(opt.DATASET_PATH, transform=None, suffix=[opt.EXT], size=opt.DATASET_SIZE)
    val_dataset = DatasetBase(opt.VALIDATION_DATASET_PATH, transform=None, suffix=[opt.EXT], size=opt.DATASET_SIZE)

    print("train", dataset_statistic(train_dataset))
    print("val", dataset_statistic(val_dataset))


if __name__ == '__main__':

    seed = 777
    torch.manual_seed(seed)
    random.seed(seed)

    opt = get_settings()

    wandb.init(project=opt.PROJECT_NAME, name=opt.CKPT_LABEL, config=dict(opt), mode=opt.WANDB_MODE)

    train_vae(opt)

    # dataset_stats(opt)
