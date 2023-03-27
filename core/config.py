import torch
from functools import lru_cache
from pydantic import BaseSettings

from .utils import log as log_utils


class Settings(BaseSettings):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    EPOCHS: int
    BATCH_SIZE: int

    LR: float  # Learning rate

    DATASET_SIZE: int = -1

    # File path
    DATASET_PATH: str
    VALIDATION_DATASET_PATH: str
    OUTPUT_PATH: str
    EXT: str = ".png"

    # Checkpoint related
    CKPT_INTERVAL: int = 100  # in batch
    CKPT_DIR: str
    CKPT_LABEL: str = ""

    RESUME: bool = True

    # Logging Related
    LOGGER_LEVEL: int = 0
    LOG_INTERVAL: int = 100  # in batch

    WANDB_MODE: str = "online"

    # Training Params
    IMG_SIZE: int = 28

    D_SIZE: int = 32
    LATENT_SIZE: int = 256

    AUG_ROTATION: int = 45
    AUG_AUTO_CONTRAST: float = 0.5
    AUG_COLOR_JITTER: int = 5

    # Metric Params
    KLD_WEIGHT: float = 1

    # Sample related
    SAMPLE_GRID_SIZE: int = 10

    # Project related
    PROJECT_NAME: str

    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings(env_path: str = ".env") -> Settings:
    opt = Settings(_env_file=env_path)  # type: ignore

    # logs settings to stdout
    logger_stdout_handler = log_utils.get_stream_handler(level=opt.LOGGER_LEVEL)
    log_utils.get_logger(handlers=[logger_stdout_handler], level=opt.LOGGER_LEVEL)

    log_utils.debug(str(opt))

    return opt
