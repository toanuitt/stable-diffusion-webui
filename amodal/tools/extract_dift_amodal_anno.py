import argparse
import os
import sys
import json

import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from tqdm import tqdm
import numpy as np

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
    parser.add_argument(
        "--anno-path",
        type=str,
        default=1,
        help="the path to json annotation file",
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


def get_annos(anno_path):
    with open(anno_path, "r") as anno_file:
        content = json.load(anno_file)

    image_dict = dict(
        [[img_info["id"], img_info] for img_info in content["images"]]
    )

    annos = []
    for anno in content["annotations"]:
        img_info = image_dict[anno["image_id"]]
        black_start = 0
        black_end = 0
        if anno["last_col"] > 0:
            black_start = anno["last_col"]
            black_end = img_info["width"]
        else:
            black_end = anno["last_col"] + 1

        annos.append(
            {
                "id": anno["id"],
                "file_name": img_info["file_name"],
                "image_height": img_info["height"],
                "black_start": black_start,
                "black_end": black_end,
            }
        )

    return annos


def main(args):
    dift = SDFeaturizer(sd_id=args.model_id)
    up_ft_indices = [0, 1, 2, 3]
    imgs_path = args.input_path
    save_path = args.output_path

    annos = get_annos(args.anno_path)

    for anno in tqdm(annos):
        img_path = os.path.join(imgs_path, anno["file_name"])
        img = np.array(Image.open(img_path))
        img[
            0 : anno["image_height"], anno["black_start"] : anno["black_end"]
        ] = [0, 0, 0]

        img_tensor = (torch.Tensor(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.permute(2, 0, 1)

        fts = dift.forward(
            img_tensor=img_tensor,
            prompt=args.prompt,
            t=args.t,
            up_ft_indices=up_ft_indices,
            ensemble_size=args.ensemble_size,
        )

        for key, value in fts.items():
            cur_folder = os.path.join(
                save_path, "t_" + str(args.t) + "_index_" + str(key)
            )
            if not os.path.exists(cur_folder):
                os.makedirs(cur_folder)

            tensor_file_name = os.path.join(
                cur_folder, f"{anno['file_name'][:-4]}_{anno['id']}_.pt"
            )
            torch.save(value.squeeze(0).cpu(), tensor_file_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
