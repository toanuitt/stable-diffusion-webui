import argparse
import os
import torch
from PIL import Image
from src.models.dift_sd import SDFeaturizer
from torchvision.transforms import PILToTensor
from tqdm import tqdm


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


def main(args):
    dift = SDFeaturizer(sd_id=args.model_id)
    up_ft_indices = [0, 1, 2, 3]
    imgs_path = args.input_path
    save_path = args.output_path
    start_index = args.start_index
    end_index = start_index + args.no_image
    img_names = [
        img_name for img_name in os.listdir(imgs_path) if is_image(img_name)
    ]

    if end_index >= len(img_names):
        end_index = len(img_names)

    chosen_img_names = img_names[start_index:end_index]

    for img_name in tqdm(chosen_img_names):
        img_path = os.path.join(imgs_path, img_name)
        img_tensor = get_image_tensor(img_path, args.img_size)
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

            tensor_file_name = os.path.join(cur_folder, img_name[:-4] + ".pt")
            torch.save(value.squeeze(0).cpu(), tensor_file_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
