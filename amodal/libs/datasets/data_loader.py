import numpy as np
import cvbase as cvb
import pycocotools.mask as maskUtils

from libs.utils import mask_to_bbox


class DatasetLoader(object):
    def __init__(self, anno_path):
        data = cvb.load(anno_path)
        self.images_info = data["images"]
        self.annot_info = data["annotations"]

        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann["regions"])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.images_info)

    def get_gt_ordering(self, imgidx):
        num = len(self.annot_info[imgidx]["regions"])
        gt_order_matrix = np.zeros((num, num), dtype=np.int)
        order_str = self.annot_info[imgidx]["depth_constraint"]
        if len(order_str) == 0:
            return gt_order_matrix
        order_str = order_str.split(",")
        for o in order_str:
            idx1, idx2 = o.split("-")
            idx1, idx2 = int(idx1) - 1, int(idx2) - 1
            gt_order_matrix[idx1, idx2] = 1
            gt_order_matrix[idx2, idx1] = -1
        return gt_order_matrix  # num x num

    def get_instance(self, idx, with_gt=False, load_occ_label=False):
        imgidx, regidx = self.indexing[idx]
        # img
        img_info = self.images_info[imgidx]
        image_fn = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]
        # region
        reg = self.annot_info[imgidx]["regions"][regidx]
        modal, bbox, category, amodal = read_MP3D(
            reg, h, w, load_occ_label=load_occ_label
        )
        return modal, bbox, category, image_fn, amodal

    def read_MP3D(self, ann, h, w, load_occ_label=False):
        assert "visible_mask" in ann.keys()  # must occluded
        m_rle = [ann["visible_mask"]]
        modal = maskUtils.decode(m_rle).squeeze()
        a_rle = [ann["segmentation"]]
        amodal = maskUtils.decode(a_rle).squeeze()
        bbox = mask_to_bbox(modal)
        category = ann["category_id"]
        return modal, bbox, category, amodal  # category as constant 1

    def get_image_instances(
        self,
        idx,
        with_gt=False,
        with_anns=False,
        ignore_stuff=False,
        load_occ_label=False,
    ):
        ann_info = self.annot_info[idx]
        img_info = self.images_info[idx]
        image_fn = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        for reg in ann_info["regions"]:
            if ignore_stuff and reg["isStuff"]:
                continue
            modal, bbox, category, amodal = self.read_MP3D(
                reg, h, w, load_occ_label=load_occ_label
            )
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            ret_amodal.append(amodal)
        if with_anns:
            return (
                np.array(ret_modal),
                ret_category,
                np.array(ret_bboxes),
                np.array(ret_amodal),
                image_fn,
                ann_info,
            )
        else:
            return (
                np.array(ret_modal),
                ret_category,
                np.array(ret_bboxes),
                np.array(ret_amodal),
                image_fn,
            )
