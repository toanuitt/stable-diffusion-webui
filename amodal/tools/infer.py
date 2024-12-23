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
from pycocotools.mask import encode, decode


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
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, "exp_path"):
        args.exp_path = os.path.dirname(args.config_path)

    tester = Tester(args)
    tester.run()


class Tester(object):
    def __init__(self):
        self.model_config = None
        self.model = None

    def prepare_model(self, model_path):
        with open("experiments\COCOA\pcnet_m\config_SDAmodal.yaml", "r") as file:
            self.model_config = yaml.safe_load(file)
        self.model = models.__dict__[self.model_config["algo"]](self.model_config, dist_model=False)
        self.model.load_state(model_path)
        self.model.switch_to("eval")

    def expand_bbox(self, bbox, height, width, enlarge_box_factor):
        centerx = bbox[0] + bbox[2] / 2.0
        centery = bbox[1] + bbox[3] / 2.0
        x_limit = bbox[2] * 1.1 if (bbox[2] * 1.1) < width else width
        y_limit = bbox[3] * 1.1 if (bbox[3] * 1.1) < height else height
        size = max(
            [
                np.sqrt(bbox[2] * bbox[3] * enlarge_box_factor),
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

    def run(self, model_path, image_root, output_root, feature_dirs):
        self.prepare_model( model_path)
        self.infer(image_root, output_root, feature_dirs)

    def infer(self, image_root, output_root, feature_dirs):
        if not os.path.exists(output_root):
            os.makedirs(output_root)


        for image_name in tqdm(os.listdir(image_root)):
            if "mask" in image_name:
                continue

            image_path = os.path.join(image_root, image_name)
            mask_path = os.path.join(
                image_root, f"{image_name.split('.')[0]}_mask.jpg"
            )

            assert os.path.exists(mask_path), f"Cannot find this {mask_path=}"

            # Load data
            modal = Image.open(mask_path)
            modal_rle = encode(np.asfortranarray(np.array(modal)))
            modal = decode(modal_rle).squeeze()

            bbox = mask_to_bbox(modal)
            #image = Image.open(image_path).convert("RGB")

            image = np.array(image_root)
            h, w = image.shape[:2]

            bbox = self.expand_bbox(bbox, h, w)

            org_src_ft_dict = infer.get_feature_from_save(feature_dirs, image_name)

            amodal_patch_pred = infer.infer_amodal(
                model=self.model,
                org_src_ft_dict=org_src_ft_dict,
                modal=modal,
                category=1,
                bbox=bbox,
                use_rgb=self.model_config["use_rgb"],
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
            amodal_name = f"amodal_mask.jpg"
            amodal_mask_path = os.path.join(output_root, amodal_name)
            amodal_mask.save(amodal_mask_path)


if __name__ == "__main__":

    model_path = "path/to/model_state.pth"
    image_root = "path/to/images"
    output_root = "path/to/output"
    feature_dirs = "path/to/features"

    # Create Tester instance and run
    tester = Tester()
    tester.run(model_path, image_root, output_root, feature_dirs)

