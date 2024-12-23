import os
import json
import numpy as np
import cv2
import copy
from tqdm import tqdm


def get_mask_area(mask, threshold):
    return np.sum(mask > threshold)


def split_mask(mask, split_percents, error_percent=0.05):
    h, w = mask.shape[:2]
    left_mask = np.zeros([h, w])
    right_mask = mask.copy()
    mask_area = get_mask_area(mask, 128)

    split_result = {}
    curr_masks = {
        "left_mask": [],
        "right_mask": [],
        "percent": 0,
        "last_col": 0,
    }

    curr_pos = 0
    curr_percent = split_percents[curr_pos]
    last_left_percent = 0
    for col in range(w):
        left_mask[0:h, col][right_mask[0:h, col] > 128] = 255
        right_mask[0:h, col] = 0

        left_mask_area = get_mask_area(left_mask, 128)
        left_mask_area_percent = left_mask_area / mask_area

        low_threshold = curr_percent - error_percent
        high_threshold = curr_percent + error_percent
        if left_mask_area_percent < low_threshold:
            continue

        if left_mask_area_percent > high_threshold:
            split_result[curr_percent] = copy.deepcopy(curr_masks)
            curr_pos += 1
            if curr_pos == len(split_percents):
                break
            curr_percent = split_percents[curr_pos]
            continue

        diff_last = abs(curr_percent - last_left_percent)
        diff_curr = abs(curr_percent - left_mask_area_percent)

        if diff_last > diff_curr:
            last_left_percent = left_mask_area_percent
            curr_masks["left_mask"] = left_mask.copy()
            curr_masks["right_mask"] = right_mask.copy()
            curr_masks["percent"] = left_mask_area_percent
            curr_masks["last_col"] = col

    return split_result


def convert_mask_to_polygon(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.squeeze().tolist() for contour in contours]
    return polygons


if __name__ == "__main__":
    image_root = "D:/HocTap/KLTN/dataset/processed/mask_create/50"
    anno_save_path = "test.json"
    split_percents = [0.3, 0.5, 0.7]

    content = {"images": [], "annotations": []}
    image_idx = 1
    anno_idx = 1
    for image_name in tqdm(os.listdir(image_root)):
        if "mask" in image_name:
            continue

        image_path = f"{image_root}/{image_name}"
        image_name_wo_ext = image_name.split(".")[0]
        mask_name = f"{image_name_wo_ext}_mask.jpg"
        mask_path = f"{image_root}/{mask_name}"

        assert os.path.exists(mask_path), f"Cannot find {mask_path=}"

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, 0)

        image_info = {
            "id": image_idx,
            "file_name": image_name,
            "width": mask.shape[1],
            "height": mask.shape[0],
        }

        content["images"].append(image_info)

        results = split_mask(mask, split_percents)

        all_polygons = convert_mask_to_polygon(mask)

        for value in results.values():
            left_polygons = convert_mask_to_polygon(value["left_mask"])
            right_polygons = convert_mask_to_polygon(value["right_mask"])

            anno = {
                "id": anno_idx,
                "image_id": image_idx,
                "last_col": value["last_col"],
                "segmentations": all_polygons,
                "visible_segmentations": left_polygons,
                "invisible_segmentations": right_polygons,
            }
            content["annotations"].append(anno)
            anno_idx += 1

            anno = {
                "id": anno_idx,
                "image_id": image_idx,
                "last_col": -value["last_col"],
                "segmentations": all_polygons,
                "visible_segmentations": right_polygons,
                "invisible_segmentations": left_polygons,
            }
            content["annotations"].append(anno)
            anno_idx += 1

        image_idx += 1

    with open(anno_save_path, "w+") as anno_file:
        json.dump(content, anno_file)
