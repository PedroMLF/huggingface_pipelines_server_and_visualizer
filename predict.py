import argparse
import json
import logging
import sys
import os

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from src.pipelines.utils import init_pipeline


def main(config: DictConfig, input_file: str):

    # Initialize pipeline
    pipeline = init_pipeline(config)

    # Get all predictions for a given input file
    predictions = {}
    for ix, line in tqdm(enumerate(open(input_file))):
        line = line.strip()
        predictions[ix] = pipeline(line)

    # Save predictions to json file
    out_json_file_path = os.path.splitext(input_file)[0] + ".json"
    with open(out_json_file_path, "w", encoding="utf-8") as out_fp:
        json.dump(predictions, out_fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    # Assert that files exist
    if not os.path.isfile(args.config_path):
        raise ValueError('config_path "{}" does not exist'.format(args.config_path))

    if not os.path.isfile(args.input_file):
        raise ValueError(('input_file "{}" does not exist').format(args.input_file))

    # Set logger
    logger = logging.getLogger("logger")
    log_formatter = logging.Formatter("[%(asctime)s] - %(name)s - %(levelname)s - %(message)s")

    # Set logger - console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Set logger - logging levels
    logger.setLevel(logging.DEBUG)

    # Read config
    config = OmegaConf.load(args.config_path)

    main(config=config, input_file=args.input_file)
