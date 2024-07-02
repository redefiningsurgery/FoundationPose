from PIL import Image
import numpy as np
from lang_sam import LangSAM

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

import time
t_start = time.time()
model = LangSAM(sam_type="vit_b")
image_pil = Image.open("./demo_data/app/rgb/0.png").convert("RGB")
text_prompt = "white-colored femur-bone"

masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25)
print(f'inference time: {time.time()-t_start}')

if len(masks) == 0:
    print(f"No objects of the '{text_prompt}' prompt detected in the image.")
else:
    masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
    for i, mask_np in enumerate(masks_np):
        mask_path = f"./demo_data/app/masks/{i}.png"
        save_mask(mask_np, mask_path)