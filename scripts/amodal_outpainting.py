import math
import numpy as np
import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import random
from modules import images, devices
from modules.processing import Processed, process_images
from modules.shared import opts, state

class EraserSetter:
    def __init__(self, config):
        self.min_overlap = config.get('min_overlap', 0.1)
        self.max_overlap = config.get('max_overlap', 0.3)
        self.min_cut_ratio = config.get('min_cut_ratio', 0.1)
        self.max_cut_ratio = config.get('max_cut_ratio', 0.9)
    
    def __call__(self, mask, image):
        # Convert inputs to numpy if needed
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Generate eraser mask
        h, w = mask.shape[:2]
        overlap = random.uniform(self.min_overlap, self.max_overlap)
        cut_ratio = random.uniform(self.min_cut_ratio, self.max_cut_ratio)
        
        # Create eraser mask
        eraser_mask = np.zeros_like(mask)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        x1 = random.randint(0, w - cut_w)
        y1 = random.randint(0, h - cut_h)
        x2 = x1 + cut_w
        y2 = y1 + cut_h
        
        eraser_mask[y1:y2, x1:x2] = 1
        
        # Apply eraser to image
        erased_image = image.copy()
        erased_image[eraser_mask == 1] = 0
        
        return erased_image, eraser_mask

def amodal_inpaint(img, mask, eraser_config=None):
    """Amodal completion inpainting based on Amodal-Completion-in-the-Wild"""
    if eraser_config is None:
        eraser_config = {
            'min_overlap': 0.1,
            'max_overlap': 0.3, 
            'min_cut_ratio': 0.1,
            'max_cut_ratio': 0.9
        }
    
    # Convert inputs to numpy arrays
    np_img = np.array(img)
    np_mask = np.array(mask)
    
    # Initialize EraserSetter
    eraser_setter = EraserSetter(eraser_config)
    
    # Apply eraser mask
    inst_erased, shift_eraser = eraser_setter(np_mask, np_img)
    
    # Return PIL Image
    return Image.fromarray(inst_erased)

class Script(scripts.Script):
    def title(self):
        return "Amodal Outpainting"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        with gr.Group():
            with gr.Row():
                info = gr.HTML("<p>Uses Amodal Completion for better outpainting results</p>")
            
            with gr.Row():
                pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128)
                mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=8)
            
            with gr.Row():
                direction = gr.CheckboxGroup(
                    label="Outpainting direction", 
                    choices=['left', 'right', 'up', 'down'],
                    value=['left', 'right', 'up', 'down']
                )

            with gr.Row():
                eraser_overlap = gr.Slider(label="Eraser overlap", minimum=0.1, maximum=0.5, step=0.05, value=0.2)
                cut_ratio = gr.Slider(label="Cut ratio", minimum=0.1, maximum=0.9, step=0.05, value=0.5)

        return [info, pixels, mask_blur, direction, eraser_overlap, cut_ratio]

    def run(self, p, _, pixels, mask_blur, direction, eraser_overlap, cut_ratio):
        initial_seed_and_info = [None, None]
        process_width = p.width
        process_height = p.height

        # Configure amodal completion params
        eraser_config = {
            'min_overlap': eraser_overlap - 0.1,
            'max_overlap': eraser_overlap + 0.1,
            'min_cut_ratio': cut_ratio - 0.2,
            'max_cut_ratio': cut_ratio + 0.2
        }

        # Setup expansion parameters
        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0
        down = pixels if "down" in direction else 0

        def expand(init, count, expand_pixels, is_left=False, is_right=False, is_top=False, is_bottom=False):
            is_horiz = is_left or is_right
            is_vert = is_top or is_bottom
            pixels_horiz = expand_pixels if is_horiz else 0
            pixels_vert = expand_pixels if is_vert else 0

            images_to_process = []
            output_images = []

            for n in range(count):
                # Create expanded image
                res_w = init[n].width + pixels_horiz
                res_h = init[n].height + pixels_vert
                process_res_w = math.ceil(res_w / 64) * 64
                process_res_h = math.ceil(res_h / 64) * 64

                img = Image.new("RGB", (process_res_w, process_res_h))
                img.paste(init[n], (pixels_horiz if is_left else 0, pixels_vert if is_top else 0))

                # Create mask for amodal completion
                mask = Image.new("RGB", (process_res_w, process_res_h), "white")
                draw = ImageDraw.Draw(mask)
                draw.rectangle((
                    expand_pixels if is_left else 0,
                    expand_pixels if is_top else 0,
                    mask.width - expand_pixels if is_right else res_w,
                    mask.height - expand_pixels if is_bottom else res_h,
                ), fill="black")

                # Apply amodal inpainting
                inpainted = amodal_inpaint(img, mask, eraser_config)
                output_images.append(inpainted)

                # Process with stable diffusion
                p.width = process_res_w
                p.height = process_res_h
                p.init_images = [inpainted]
                p.image_mask = mask
                p.latent_mask = mask

                output_images = process_images(p).images
                
                # Crop to target size
                for i in range(len(output_images)):
                    output_images[i] = output_images[i].crop((0, 0, res_w, res_h))

            return output_images

        # Process batches
        batch_count = p.n_iter
        batch_size = p.batch_size
        p.n_iter = 1
        state.job_count = batch_count * sum(1 for x in [left, right, up, down] if x > 0)
        all_processed_images = []

        for i in range(batch_count):
            imgs = [p.init_images[0]] * batch_size
            state.job = f"Batch {i + 1} out of {batch_count}"

            if left > 0:
                imgs = expand(imgs, batch_size, left, is_left=True)
            if right > 0:
                imgs = expand(imgs, batch_size, right, is_right=True)
            if up > 0:
                imgs = expand(imgs, batch_size, up, is_top=True)
            if down > 0:
                imgs = expand(imgs, batch_size, down, is_bottom=True)

            all_processed_images += imgs

        # Create grid and return results
        grid = images.image_grid(all_processed_images)
        return Processed(p, all_processed_images, initial_seed_and_info[0], initial_seed_and_info[1])
