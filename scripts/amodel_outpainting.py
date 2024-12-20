import math
import numpy as np
import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw
from modules import images, devices
from modules.processing import Processed, process_images
from modules.shared import opts, state
from amodal.dift.src.models.dift_sd import SDFeaturizer
from amodal.dift.src.utils.visualization import Demo
def get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05):
    # helper fft routines that keep ortho normalization and auto-shift before and after fft
    def _fft2(data):
        if data.ndim > 2:  # has channels
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
                out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
        else:  # one channel
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
            out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

        return out_fft

    def _ifft2(data):
        if data.ndim > 2:  # has channels
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
                out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
        else:  # one channel
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
            out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

        return out_ifft

    def _get_gaussian_window(width, height, std=3.14, mode=0):
        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        window = np.zeros((width, height))
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            if mode == 0:
                window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
            else:
                window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (std / 3.14)  # hey wait a minute that's not gaussian

        return window

    def _get_masked_window_rgb(np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        for c in range(3):
            np_mask_rgb[:, :, c] = hardened[:]
        return np_mask_rgb

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    # create a generator with a static seed to make outpainting deterministic / only follow global seed
    rng = np.random.default_rng(0)

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = rng.random((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist ** noise_q) * src_phase  # perform the actual shaping

    brightness_variation = 0.  # color_variation # todo: temporarily tying brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1., contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb

    matched_noise = shaped_noise[:]

    return np.clip(matched_noise, 0., 1.)


class Script(scripts.Script):
    def title(self):
        return "Amodal Outpainting" 

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        info = gr.HTML("<p>Uses amodal segmentation to guide outpainting</p>")
        pixels = gr.Slider(label="Pixels to expand", minimum=8, maximum=256, step=8, value=128)
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8) 
        direction = gr.CheckboxGroup(label="Direction", choices=['left', 'right', 'up', 'down'])
        noise_q = gr.Slider(label="Fall-off exponent", minimum=0.0, maximum=4.0, step=0.01, value=1.0)
        color_variation = gr.Slider(label="Color variation", minimum=0.0, maximum=1.0, step=0.01, value=0.05)

        return [info, pixels, mask_blur, direction, noise_q, color_variation]

    def run(self, p, _, pixels, mask_blur, direction, noise_q, color_variation):
        # Get initial image
        init_img = p.init_images[0]
        
        # Extract amodal mask using DIFT
        dift = SDFeaturizer(device=devices.device)
        demo = Demo()
        features = dift.extract_features(init_img)
        amodal_mask = demo.process_features(features)
        
        # Convert to PIL Image
        amodal_mask = Image.fromarray((amodal_mask * 255).astype(np.uint8))
        
        # Setup outpainting parameters 
        p.inpaint_full_res = False
        p.inpainting_fill = 1
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0 
        down = pixels if "down" in direction else 0

        # Calculate blur amounts
        mask_blur_x = mask_blur if (left > 0 or right > 0) else 0
        mask_blur_y = mask_blur if (up > 0 or down > 0) else 0
        p.mask_blur_x = mask_blur_x * 4
        p.mask_blur_y = mask_blur_y * 4

        # Calculate target dimensions
        target_w = math.ceil((init_img.width + left + right) / 64) * 64
        target_h = math.ceil((init_img.height + up + down) / 64) * 64

        # Adjust padding to maintain aspect ratio
        if left > 0:
            left = left * (target_w - init_img.width) // (left + right)
        if right > 0:
            right = target_w - init_img.width - left
        if up > 0:
            up = up * (target_h - init_img.height) // (up + down)
        if down > 0:
            down = target_h - init_img.height - up

        # Create expanded canvas
        img = Image.new("RGB", (target_w, target_h))
        img.paste(init_img, (left, up))

        # Create combined mask
        mask = Image.new("L", (img.width, img.height), "white")
        draw = ImageDraw.Draw(mask)
        
        # Paste amodal mask
        mask.paste(amodal_mask, (left, up))
        
        # Add directional masking
        draw.rectangle((
            left + (mask_blur_x * 2 if left > 0 else 0),
            up + (mask_blur_y * 2 if up > 0 else 0), 
            mask.width - right - (mask_blur_x * 2 if right > 0 else 0),
            mask.height - down - (mask_blur_y * 2 if down > 0 else 0)
        ), fill="black")

        # Create noise for outpainting region 
        np_img = (np.asarray(img) / 255.0).astype(np.float64)
        np_mask = (np.asarray(mask) / 255.0).astype(np.float64) 
        noised = get_matched_noise(np_img, np_mask, noise_q, color_variation)
        img = Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8))

        # Set up processing
        p.init_images = [img]
        p.image_mask = mask
        p.width = target_w
        p.height = target_h

        # Process image
        proc = process_images(p)

        # Save results
        if opts.samples_save:
            images.save_image(proc.images[0], p.outpath_samples, "", proc.seed, p.prompt, opts.samples_format, info=proc.info, p=p)

        return Processed(p, proc.images, proc.seed, proc.info)