import logging


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(handler)
