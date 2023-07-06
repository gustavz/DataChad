import logging
import sys

logger = logging.getLogger(__name__)


def configure_logger(debug: int = 0) -> None:
    # boilerplate code to enable logging in the streamlit app console
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter("%(name)s :: %(levelname)s :: %(message)s")

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False


configure_logger(0)
