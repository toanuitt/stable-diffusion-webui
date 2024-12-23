import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

class SDModelHandler:
    def __init__(self):
        self.model_path = "/kaggle/input/inpainting/realisticVisionV60B1_v51VAE-inpainting.safetensors"
        self.input_dir = "inputs"
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        ).to("cuda")

    def load_images(self, image_name):
        image_path = os.path.join(self.input_dir, f"{image_name}.jpg")
        mask_path = os.path.join(self.input_dir, f"{image_name}_mask.jpg")
        
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')
        
        return image, mask

    def generate_image(self, prompt, image, mask):
        # Convert mask to proper format
        mask = np.array(mask)
        mask = mask > 127  # Convert to binary mask
        mask = Image.fromarray(mask.astype(np.uint8) * 255)

        # Generate image with mask guidance
        image = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        output_path = os.path.join(self.output_dir, f"generated_{len(os.listdir(self.output_dir))}.png")
        image.save(output_path)
        return output_path

    def inference(self, image_name, prompt):
        if not hasattr(self, 'pipe'):
            self.load_model()
            
        image, mask = self.load_images(image_name)
        output_path = self.generate_image(prompt, image, mask)
        return output_path

# Usage
if __name__ == "__main__":
    handler = SDModelHandler()
    result = handler.inference(
        image_name="/kaggle/input/coco-300-input/300/000000000071_1",  # Will look for "image.jpg" and "image_mask.jpg"
        prompt=""
    )
    print(f"Generated image saved at: {result}")