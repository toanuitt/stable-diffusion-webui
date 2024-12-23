import argparse
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm

sys.path.append(".")
from libs.dift.dift_sd import SDFeaturizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="""extract dift from input image, and save it as torch tensor,
                    in the shape of [c, h, w]."""
    )
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=[768, 768],
        help="""in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.""",
    )
    parser.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-2-1",
        type=str,
        help="model_id of the diffusion model in huggingface",
    )
    parser.add_argument(
        "--t",
        default=181,
        type=int,
        help="time step for diffusion, choose from range [0, 1000]",
    )
    parser.add_argument(
        "--prompt",
        default="",
        type=str,
        help="prompt used in the stable diffusion",
    )
    parser.add_argument(
        "--ensemble-size",
        default=8,
        type=int,
        help="number of repeated images in each batch used to get features",
    )
    parser.add_argument(
        "--input-path", type=str, help="path to the input image file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="dift.pt",
        help="path to save the output features as torch tensor",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="the first index of first image to take feature",
    )
    parser.add_argument(
        "--no-image",
        type=int,
        default=1,
        help="the number of images to take feature",
    )
    args = parser.parse_args()
    return args


def is_image(file_name: str) -> bool:
    file_name = file_name.lower()
    return file_name.endswith(".jpg") or file_name.endswith(".png")


def get_image_tensor(image_path: str, image_size: list):
    img = Image.open(image_path).convert("RGB")
    if image_size[0] != -1:
        img = img.resize(image_size)

    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2

    return img_tensor


def extract(img_tensor):
    dift = SDFeaturizer(sd_id="stabilityai/stable-diffusion-2-1")
    up_ft_indices = [0, 1, 2, 3]
    save_path = "data/dift_features"

    # Process single image
    fts = dift.forward(
        img_tensor=img_tensor,
        prompt="",
        t=181,
        up_ft_indices=up_ft_indices,
    )
    
    # Save features
    save_name = 'features.pth'
    save_file = os.path.join(save_path, save_name)
    torch.save(fts, save_file)
    return fts

