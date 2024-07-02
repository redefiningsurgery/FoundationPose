### conda activate abed_env
### cd FoundationPose/
###  python rgbcad2gif.py
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform,
                                RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
                                SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,)
import imageio
from tqdm import tqdm

dataset_name = 'bone'
cad_im_size = 386

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

verts, faces_idx, _ = load_obj(f"./demo_data/{dataset_name}/mesh/textured_simple.obj")
faces = faces_idx.verts_idx
verts_rgb = torch.ones_like(verts)[None]
textures = TexturesVertex(verts_features=verts_rgb.to(device))

object_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

cameras = FoVPerspectiveCameras(device=device)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
# raster_settings = RasterizationSettings(
#     image_size=cad_im_size, 
#     blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
#     faces_per_pixel=100, 
# )

# silhouette_renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras, 
#         raster_settings=raster_settings
#     ),
#     shader=SoftSilhouetteShader(blend_params=blend_params)
# )

raster_settings = RasterizationSettings(
    image_size=cad_im_size, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

distance = 1
elevation = 50.0
azimuth = 0.0

R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

images = []

for frame_number in tqdm(range(619), desc="Processing frames"):
    with open(f"./demo_data/{dataset_name}/ob_in_cam/{frame_number:06d}.txt", 'r') as file:
        data_string = file.read()
    rgb = cv2.imread(f"./demo_data/{dataset_name}/rgb/{frame_number:06d}.png")

    new_values = np.fromstring(data_string, sep=" ").reshape(4, 4)
    new_tensor = torch.tensor(new_values, dtype=torch.float32)
    new_tensor = new_tensor.to(device)
    R[0, :3, :3] = torch.inverse(new_tensor[:3, :3])
    new_tensor[:3, 3] = 10*new_tensor[:3, 3]
    T[0,0], T[0,1], T[0,2] = new_tensor[0, 3], new_tensor[1, 3], new_tensor[2, 3]
    image_ref = phong_renderer(meshes_world=object_mesh, R=R, T=T)
    image_ref = image_ref.cpu().numpy()
    img = np.flip(np.flip(image_ref.squeeze(), axis=0), axis=1)
    img_rgb = 255*img[:, :, :3]
    rgb[0:cad_im_size, 0:cad_im_size, :] = img_rgb
    rgb = cv2.resize(rgb, (480, 270))
    images.append(rgb)

fps = 20
gif_path=f'demo_data/{dataset_name}/{dataset_name}_cad.gif'
print('Saving GIF ...')
imageio.mimsave(gif_path, images, fps=fps)