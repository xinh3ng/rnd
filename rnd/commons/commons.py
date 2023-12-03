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
