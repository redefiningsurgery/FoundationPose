from PIL import Image
import numpy as np
from lang_sam import LangSAM
import os
from tqdm import tqdm

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

# Initialize the model
model = LangSAM(sam_type="vit_b")
text_prompt = "body"

# Input and output directories
input_dir = "./lsam_frames/video"
output_dir = "./lsam_frames/masks"
os.makedirs(output_dir, exist_ok=True)

# Get list of all images in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

# Process each image
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_dir, image_file)
    image_pil = Image.open(image_path).convert("RGB")
    
    # Perform prediction
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25)
    
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in {image_file}.")
    else:
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
        for i, mask_np in enumerate(masks_np):
            mask_path = os.path.join(output_dir, image_file)
            save_mask(mask_np, mask_path)
