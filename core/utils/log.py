import logging
import sys

from logging import Logger, Handler
from typing import IO, List, Optional

DEFAULT_LOGGER_NAME = "default-logger"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

CURRENT_LOGGERS = {}

root = logging.getLogger()
root.setLevel(logging.DEBUG)


def get_stream_handler(stream: IO[str] = sys.stdout, level: int = logging.INFO, log_format: str = DEFAULT_LOG_FORMAT) -> Handler:
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level=level)

    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    return handler


def get_logger(name: str = DEFAULT_LOGGER_NAME, handlers: List[Handler] = [], level: int = logging.INFO) -> Logger:
    global CURRENT_LOGGERS

    logger = CURRENT_LOGGERS.get(name)

    if isinstance(logger, Logger):
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in handlers:
        logger.addHandler(handler)

    CURRENT_LOGGERS[name] = logger

    return logger


def info(msg: str, json: Optional[dict] = None, logger_name: str = DEFAULT_LOGGER_NAME):
    logger = get_logger(logger_name, level=logging.INFO)

    if isinstance(json, dict):
        logger.info(msg, extra={'data': json})
    else:
        logger.info(msg)


def debug(msg: str, json: Optional[dict] = None, logger_name: str = DEFAULT_LOGGER_NAME):
    logger = get_logger(logger_name, level=logging.DEBUG)

    if isinstance(json, dict):
        logger.debug(msg, extra={'data': json})
    else:
        logger.debug(msg)


def warning(msg: str, json: Optional[dict] = None, logger_name: str = DEFAULT_LOGGER_NAME):
    logger = get_logger(logger_name, level=logging.WARNING)

    if isinstance(json, dict):
        logger.warning(msg, extra={'data': json})
    else:
        logger.warning(msg)


def error(msg: str, json: Optional[dict] = None, logger_name: str = DEFAULT_LOGGER_NAME):
    logger = get_logger(logger_name, level=logging.ERROR)

    if isinstance(json, dict):
        logger.error(msg, extra={'data': json})
    else:
        logger.error(msg)


def compose_msg(msg: str, func_name: str = "unknown") -> str:
    return "[{:20}] {}".format(func_name, msg)
