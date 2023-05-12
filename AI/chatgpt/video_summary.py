"""

# Links

"""
import json
from totepyutils.generic import create_logger

logger = create_logger(__name__, level="info")


def main(**kwargs):
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=100)
    args = vars(parser.parse_args())
    logger.info("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    logger.info("\nALL DONE!\n")
