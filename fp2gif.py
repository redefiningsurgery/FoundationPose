import os
import numpy as np
import cv2
import imageio
from tqdm import tqdm
import argparse

def K_reader(dataset_name):
    txt_path=f'demo_data/{dataset_name}/cam_K.txt'
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    numbers = []
    for line in lines:
        numbers.extend([float(num) for num in line.split() if num.strip()])
    intrinsics_matrix = np.array(numbers).reshape(3, 3)
    return intrinsics_matrix

def T_reader(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    numbers = []
    for line in lines:
        numbers.extend([float(num) for num in line.split() if num.strip()])
    transformation_matrix = np.array(numbers).reshape(4, 4)
    return transformation_matrix

def rgb_reader(image_path, show=False):
    image = cv2.imread(image_path)
    if show:
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image

def project2image(point, K):
    projected_point = K@point
    x = projected_point[0] / projected_point[2]
    y = projected_point[1] / projected_point[2]
    return (int(x), int(y))  # int format as pixel number

def axis_visualizer(image, K, T, axis_length=0.15, axis_thickness=5):
    P, R = T[:3, 3], T[:3, :3]
    X = P + axis_length * R @ np.array([1,0,0])
    Y = P + axis_length * R @ np.array([0,1,0])
    Z = P + axis_length * R @ np.array([0,0,1])
    O = project2image(point=P, K=K)
    x = project2image(point=X, K=K)
    y = project2image(point=Y, K=K)
    z = project2image(point=Z, K=K)
    cv2.line(image, O, x, color=(0,0,255), thickness=axis_thickness)
    cv2.line(image, O, y, color=(0,255,0), thickness=axis_thickness)
    cv2.line(image, O, z, color=(255,0,0), thickness=axis_thickness)
    return image


def save_gif(dataset_name, num_files=None, fps=20):
    gif_path=f'demo_data/{dataset_name}/{dataset_name}.gif'
    images = []
    K = K_reader(dataset_name)
    image_files = [f for f in os.listdir(f'demo_data/{dataset_name}/rgb') if f.endswith('.png')]
    
    image_indices = sorted([int(f.split('.')[0]) for f in image_files])
    
    if num_files is not None:
        image_indices = image_indices[:1800]#num_files-1]
    
    for index in tqdm(image_indices, desc='Generating GIF'):
        doubled_index = f"{index:06d}"
        
        T = T_reader(txt_path=f'demo_data/{dataset_name}/ob_in_cam/{doubled_index}.txt')
        image = rgb_reader(image_path=f'demo_data/{dataset_name}/rgb/{doubled_index}.png', show=False) 
        visualized_image = axis_visualizer(image=image, K=K, T=T)
        
        image_rgb = visualized_image
        # image_rgb = cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (480, 270))
        images.append(image_rgb)
    
    print('Saving GIF ...')
    imageio.mimsave(gif_path, images, fps=fps)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate a GIF from images and transformation matrices.')
    parser.add_argument('--dataset_name', type=str, default='mustard0', help='The name of the dataset')
    parser.add_argument('--num_files', type=int, default=5000, help='Number of files to include in the GIF.')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for the GIF.')
    
    args = parser.parse_args()
    
    save_gif(dataset_name=args.dataset_name, num_files=args.num_files, fps=args.fps)
