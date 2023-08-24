import logging
import sys


def create_logger(level="DEBUG"):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(level)
    # if no streamhandler present, add one
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter("%(name)s :: %(levelname)s :: %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


logger = create_logger()
