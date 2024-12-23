import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import sys

sys.path.append(".")
from libs.datasets import reader
from libs import models
import libs.utils.inference as infer
from libs import utils
from tqdm import tqdm
import torch

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--image-root", required=True, type=str)
    parser.add_argument("--feature-dirs", required=True, type=str)
    parser.add_argument("--output-root", default=None, type=str)
    parser.add_argument("--order-th", default=0.1, type=float)
    parser.add_argument("--amodal-th", default=0.2, type=float)
    parser.add_argument("--dilate-kernel", default=0, type=int)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, "exp_path"):
        args.exp_path = os.path.dirname(args.config)

    tester = Tester(args)
    tester.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
