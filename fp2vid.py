import cv2
import glob
import os
from tqdm import tqdm

def process_images_to_video(dataset_name, frame_width, frame_height):
    input_path = f'demo_data/{dataset_name}/track_vis/'
    output_path = f'demo_data/{dataset_name}/fp_{dataset_name}.mp4'
    
    images = sorted(glob.glob(os.path.join(input_path, "*.png")))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 60.0, (frame_width, frame_height))
    
    error_images = []  # List to store names of images that cause exceptions
    
    for img_path in tqdm(images, desc="Processing images"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Image cannot be read: {img_path}")
            img = cv2.resize(img, (frame_width, frame_height))
            out.write(img)
        except Exception as e:
            error_images.append(img_path)
            print(f"Error processing {img_path}: {e}")
    
    out.release()
    
    if error_images:
        print("Errors occurred with the following images:")
        for error_img in error_images:
            print(error_img)

# Call the function with desired parameters
process_images_to_video('cutie', 1440, 810)