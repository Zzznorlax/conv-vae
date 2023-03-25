import os
import random
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from core.config import Settings, get_settings
from core.model import ConvVAE
from core.dataset import DatasetBase
from core.metric import elbo_loss
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

    train_dataset = DatasetBase(opt.DATASET_PATH, transform=aug_transforms, suffix=[opt.EXT], size=opt.DATASET_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=opt.BATCH_SIZE, shuffle=True)

    val_dataset = DatasetBase(opt.VALIDATION_DATASET_PATH, transform=None, suffix=[opt.EXT], size=opt.DATASET_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=opt.BATCH_SIZE, shuffle=False)

    model = ConvVAE(in_ch=3, input_size=28, latent_dim=opt.LATENT_SIZE, d_size=opt.D_SIZE)

    optimizer = Adam(model.parameters(), lr=opt.LR)

    # custom loss function
    def _loss_func(x, y, mu, logvar):

        return elbo_loss(x, y, mu, logvar)

    # loads state from previous checkpoint
    if opt.RESUME:
        ckpt = get_latest_ckpt(opt.CKPT_DIR)
        if ckpt:
            state_dict = torch.load(ckpt)
            model.load_state_dict(state_dict['model_state'])
            cur_itrs = state_dict['cur_itrs']
            best_loss = state_dict['best_loss']
            optimizer.load_state_dict(state_dict['optimizer_state'])

    device = opt.DEVICE
    model.to(device)

    wandb.watch(model, log="all", log_freq=100)

    best_loss = 1e10
    cur_itrs = 0
    for epoch in range(opt.EPOCHS):

        for batch_idx, img in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()

            img = img.to(device)

            x, mu, logvar = model(img)

            loss = _loss_func(img, x, mu, logvar)

            loss.backward()
            optimizer.step()

            logger.info("epoch ({}/{}), batch {}, Training Loss: {}".format(epoch + 1, opt.EPOCHS, batch_idx + 1, loss))

            if (batch_idx + 1) % opt.LOG_INTERVAL == 0:

                model.eval()
                with torch.no_grad():
                    for val_batch_idx, val_img in enumerate(val_loader):

                        val_img = val_img.to(device)

                        val_x, val_mu, val_logvar = model(val_img)

                        val_loss = _loss_func(val_img, val_x, val_mu, val_logvar)

                        logger.info("epoch ({}/{}), batch {}, Validation Loss: {}".format(epoch + 1, opt.EPOCHS, val_batch_idx + 1, val_loss))

                        wandb.log({
                            "val_loss": val_loss,
                            "train_loss": loss,
                        })

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

        cur_itrs += opt.BATCH_SIZE

    logger.info("Training finished")


if __name__ == '__main__':

    seed = 777
    torch.manual_seed(seed)
    random.seed(seed)

    opt = get_settings()

    wandb.init(project=opt.PROJECT_NAME, name=opt.CKPT_LABEL, config=dict(opt), mode=opt.WANDB_MODE)

    train_vae(opt)
