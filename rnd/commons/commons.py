import logging
from logging.handlers import RotatingFileHandler
import time


def retry(n_trials: int, sleep: int = 1, *exception_types):
    """Retry a function a few times with a sleep in the middle"""

    def try_fn(func, *args, **kwargs):
        for n in range(n_trials):
            if n == n_trials - 1:  # This is the last try
                return func(*args, **kwargs)

            try:
                return func(*args, **kwargs)
            except exception_types or Exception as e:
                logger = create_logger("retry", level="info")
                logger.warning(f"Trial: {n} failed with exception {e}. Trying again after a {sleep}-second sleep")
                time.sleep(sleep)

    return try_fn


def create_logger(
    name: str,
    level: str = "info",
    fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    add_file_handler: bool = False,
    logfile: str = "/tmp/tmp.log",
):
    """Create a formatted logger

    Examples:
        logger = create_logger(__name__, level="debug")
        logger.info("Hello World")
    """
    level = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARN, "error": logging.ERROR}.get(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if fmt == "json":
        log_formatter = jsonlogger.JsonFormatter()
    else:
        log_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Print on console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

    # Print in a log file
    if add_file_handler:
        if "s3://" in logfile:
            logfile = "/tmp/tmp.log"  # Put in local first

        th = RotatingFileHandler(logfile, mode="a", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        th.setFormatter(log_formatter)
        logger.addHandler(th)

    return logger
