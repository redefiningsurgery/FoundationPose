import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def read_transformation_matrices(folder_path):
    file_paths = sorted(glob.glob(os.path.join(folder_path, '*.txt')))
    data_matrix = []

    for file_path in file_paths:
        matrix = np.loadtxt(file_path)
        position = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        orientation = rotation.as_euler('xyz', degrees=True)
        data_matrix.append(np.concatenate((position, orientation)))

    return np.array(data_matrix)

def create_animation(data_matrix, fps=10, frame2propagate=None):
    if frame2propagate is not None:
        data_matrix = data_matrix[:frame2propagate]
    
    # Extract positions and calculate axes limits
    positions = data_matrix[:, :3]
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    # Apply a 20% margin
    x_margin = (x_max - x_min) * 0.2
    y_margin = (y_max - y_min) * 0.2
    z_margin = (z_max - z_min) * 0.2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)

    def update_plot(num, ax, fig, pbar):
        ax.clear()
        xs, ys, zs = positions[:num+1].T
        ax.plot(xs, ys, zs, marker='o', label='Position Trajectory')
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        pbar.update(1)
    
    pbar = tqdm(total=len(positions), desc="Generating Animation")
    ani = FuncAnimation(fig, update_plot, frames=len(positions), fargs=(ax, fig, pbar),
                        interval=1000/fps, repeat=False)

    ani.save('trajectory_animation.mp4', writer='ffmpeg', dpi=200)
    plt.close(fig)
    pbar.close()
    print("Animation saved as 'trajectory_animation.mp4'.")

# Example usage
dataset_name = 'cutie'
folder_path = f'demo_data/{dataset_name}/ob_in_cam/'
print('Reading text files and saving 6D poses as a matrix...')
data_matrix = read_transformation_matrices(folder_path)
create_animation(data_matrix, fps=30, frame2propagate=100)  # Set frame2propagate here
