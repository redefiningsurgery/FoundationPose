from PIL import Image
import numpy as np
import os
from lang_sam import LangSAM
from tqdm import tqdm

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

dataset_name = "bone_4k"  # Replace this with your dataset name

# Directory paths
rgb_dir = f"./demo_data/{dataset_name}/rgb"
masks_dir = f"./demo_data/{dataset_name}/masks"

# Create masks directory if it doesn't exist
os.makedirs(masks_dir, exist_ok=True)

# Initialize the model
model = LangSAM(sam_type="vit_b")
text_prompt = "white-colored femur-bone"

# List all .png files in the RGB directory
image_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]

# Process each image in the RGB directory with a progress bar
for filename in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(rgb_dir, filename)
    image_pil = Image.open(image_path).convert("RGB")

    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25)

    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image {filename}.")
    else:
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
        for i, mask_np in enumerate(masks_np):
            mask_filename = os.path.join(masks_dir, filename)
            save_mask(mask_np, mask_filename)
