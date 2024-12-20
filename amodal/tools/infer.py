import argparse
import yaml
import os
import numpy as np
from PIL import Image
import sys

from tqdm import tqdm

# import torch
import torchvision.transforms as transforms

sys.path.append(".")
from libs import models
from libs.utils import inference as infer
from libs.utils.data_utils import mask_to_bbox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--image-root", required=True, type=str)
    parser.add_argument("--feature-dirs", required=True, type=str)
    parser.add_argument("--output-root", default=None, type=str)
    # parser.add_argument("--order-method", required=True, type=str)
    # parser.add_argument("--amodal-method", required=True, type=str)
    parser.add_argument("--order-th", default=0.1, type=float)
    parser.add_argument("--amodal-th", default=0.2, type=float)
    parser.add_argument("--dilate-kernel", default=0, type=int)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, "exp_path"):
        args.exp_path = os.path.dirname(args.config_path)

    tester = Tester(args)
    tester.run()


class Tester(object):
    def __init__(self, args):
        self.args = args

    def prepare_model(self):
        self.model = models.__dict__[self.args.model["algo"]](
            self.args.model, dist_model=False
        )
        self.model.load_state(self.args.model_path)
        self.model.switch_to("eval")

    def expand_bbox(self, bbox, height, width):
        centerx = bbox[0] + bbox[2] / 2.0
        centery = bbox[1] + bbox[3] / 2.0
        x_limit = bbox[2] * 1.1 if (bbox[2] * 1.1) < width else width
        y_limit = bbox[3] * 1.1 if (bbox[3] * 1.1) < height else height
        size = max(
            [
                np.sqrt(bbox[2] * bbox[3] * self.args.data["enlarge_box"]),
                x_limit,
                y_limit,
            ]
        )
        new_bbox = [
            int(centerx - size / 2.0),
            int(centery - size / 2.0),
            int(size),
            int(size),
        ]
        return np.array(new_bbox)

    def run(self):
        self.prepare_model()
        self.infer()

    def infer(self):
        if not os.path.exists(self.args.output_root):
            os.makedirs(self.args.output_root)

        self.args.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    self.args.data["data_mean"], self.args.data["data_std"]
                ),
            ]
        )

        for image_name in tqdm(os.listdir(self.args.image_root)):
            if "mask" in image_name:
                continue

            image_path = os.path.join(self.args.image_root, image_name)
            mask_path = os.path.join(
                self.args.image_root, f"{image_name.split('.')[0]}_mask.jpg"
            )

            assert os.path.exists(mask_path), f"Cannot find this {mask_path=}"

            # data
            modal = Image.open(mask_path)
            modal = np.array(modal)

            bbox = mask_to_bbox(modal)
            image = Image.open(image_path).convert("RGB")

            image = np.array(image)
            h, w = image.shape[:2]

            bbox = self.expand_bbox(bbox, h, w)

            org_src_ft_dict = infer.get_feature_from_save(
                self.args.feature_dirs, image_name
            )

            amodal_patch_pred = infer.infer_amodal(
                model=self.model,
                org_src_ft_dict=org_src_ft_dict,
                modal=modal,
                category=1,
                bbox=bbox,
                use_rgb=self.args.model["use_rgb"],
                input_size=512,
                min_input_size=16,
                interp="nearest",
            )

            amodal_pred = infer.recover_mask(
                mask=amodal_patch_pred,
                bbox=bbox,
                height=h,
                width=w,
                interp="linear",
            )

            amodal_mask = Image.fromarray(amodal_pred * 255).convert("RGB")
            amodal_name = f"{image_name.split('.')[0]}_amodal_mask.jpg"
            amodal_mask_path = os.path.join(self.args.output_root, amodal_name)
            amodal_mask.save(amodal_mask_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
