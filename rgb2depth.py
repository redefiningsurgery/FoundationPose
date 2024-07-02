import cv2
import torch
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm

def find_center_of_mask(mask):
    moments = cv2.moments(mask)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    return cX, cY

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

dataset_name = "bone_4k"
rgb_folder = f"demo_data/{dataset_name}/rgb"
depth_folder = f"demo_data/{dataset_name}/depth"
depth_lidar_folder = f"demo_data/{dataset_name}/depth_lidar"
masks_folder = f"demo_data/{dataset_name}/masks"

# Create depth folder if it doesn't exist
os.makedirs(depth_folder, exist_ok=True)

# Get all PNG images in the rgb folder
image_paths = glob.glob(os.path.join(rgb_folder, "*.png"))

# Process images with a progress bar
for img_path in tqdm(image_paths, desc="Processing images"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Load corresponding mask and depth_lidar images
    base_name = os.path.basename(img_path)
    mask_path = os.path.join(masks_folder, base_name)
    depth_lidar_path = os.path.join(depth_lidar_folder, base_name)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    depth_lidar = cv2.imread(depth_lidar_path, cv2.IMREAD_UNCHANGED)

    # Find the center of the mask
    cX, cY = find_center_of_mask(mask)

    # Adjust the coordinates for the lower resolution of depth_lidar
    scale_x = depth_lidar.shape[1] / img.shape[1]
    scale_y = depth_lidar.shape[0] / img.shape[0]
    cX_lidar = int(cX * scale_x)
    cY_lidar = int(cY * scale_y)

    # Get the depth values
    depth_abs = depth_lidar[cY_lidar, cX_lidar] ### make sure about the aspects of the cameras
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # Get the relative depth value at the center of the mask
    depth_rel = output[cY, cX]

    # Adjust the relative depth map
    if depth_rel != 0:  # Avoid division by zero
        output = output * (depth_abs / depth_rel)

    # Convert the adjusted depth map to uint16 to preserve depth values
    output_uint16 = output.astype(np.uint16)

    # Save the depth map using Pillow
    depth_map_path = os.path.join(depth_folder, base_name)
    depth_map_image = Image.fromarray(output_uint16)
    depth_map_image.save(depth_map_path)

print("Depth maps have been saved successfully.")
