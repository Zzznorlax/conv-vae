import os
import random
import torch
import wandb
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

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

        for batch_idx, img in enumerate(train_loader):

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

            # performs validation and logs to wandb
            if (batch_idx + 1) % opt.LOG_INTERVAL == 0:

                val_cum_batch = 0
                val_total_loss = 0
                val_total_kld = 0
                model.eval()
                with torch.no_grad():
                    for _, val_img in enumerate(val_loader):

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

                logger.info("epoch ({}/{}), iteration {}, Validation Loss: {}".format(epoch + 1, opt.EPOCHS, cur_itrs + 1, val_avg_loss))
                logger.info("epoch ({}/{}), iteration {}, Training Loss: {}".format(epoch + 1, opt.EPOCHS, cur_itrs + 1, train_avg_loss))

                wandb.log({
                    "val_kld": val_avg_kld,
                    "val_loss": val_avg_loss,
                    "train_kld": train_avg_kld,
                    "train_loss": train_avg_loss,
                    "iteration": cur_itrs + 1,
                })
                train_total_loss = 0
                train_total_kld = 0
                cum_batch = 0

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

        # generates samples and logs image to wandb
        sample_loader = DataLoader(val_dataset, batch_size=opt.SAMPLE_GRID_SIZE**2)

        samples = None
        input_imgs = None
        for _, img in enumerate(sample_loader):

            img = img.to(opt.DEVICE)
            recon_imgs, _, _ = model(img)
            samples = recon_imgs.cpu().detach().numpy()
            input_imgs = img.cpu().detach().numpy()

            break

        if samples is not None:
            # Rescale pixel values from [0, 1] to [0, 255]
            samples = (samples * 255).astype(np.uint8)

            # Reshape samples to a 10 by 10 grid of images
            samples = samples.reshape(opt.SAMPLE_GRID_SIZE, opt.SAMPLE_GRID_SIZE, opt.IMG_SIZE, opt.IMG_SIZE)
            samples = np.transpose(samples, (0, 2, 1, 3))
            samples = samples.reshape(opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE, opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE)

        if input_imgs is not None:
            # Rescale pixel values from [0, 1] to [0, 255]
            input_imgs = (input_imgs * 255).astype(np.uint8)

            # Reshape input images to a 10 by 10 grid of images
            input_imgs = input_imgs.reshape(opt.SAMPLE_GRID_SIZE, opt.SAMPLE_GRID_SIZE, opt.IMG_SIZE, opt.IMG_SIZE)
            input_imgs = np.transpose(input_imgs, (0, 2, 1, 3))
            input_imgs = input_imgs.reshape(opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE, opt.SAMPLE_GRID_SIZE * opt.IMG_SIZE)

        wandb.log({
            "epoch": epoch + 1,
            "samples": wandb.Image(samples, caption="Samples"),
            "inputs": wandb.Image(input_imgs, caption="Inputs"),
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
