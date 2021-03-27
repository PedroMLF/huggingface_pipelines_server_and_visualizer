import argparse
import logging
import sys

from omegaconf import OmegaConf

from src.pipelines.utils import init_pipeline


def main(config):

    # Initialize pipeline
    pipeline = init_pipeline(config)

    # TODO: Add typer interactive client?
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    # Set logger
    logger = logging.getLogger("logger")
    log_formatter = logging.Formatter(
        "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    )

    # Set logger - console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Set logger - logging levels
    logger.setLevel(logging.DEBUG)

    # Read config
    config = OmegaConf.load(args.config_path)

    main(config)
