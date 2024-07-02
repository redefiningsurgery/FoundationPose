from estimater import *
from datareader import *
import argparse
import time


def get_bounding_box_coordinates(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    x_min, y_min = mask.shape[1], mask.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    margin = 50
    return max(y_min - margin, 0), min(y_max + margin, mask.shape[0]), max(x_min - margin, 0), min(x_max + margin, mask.shape[1])



if __name__=='__main__':

  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_name = "cutie"
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/{dataset_name}/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/{dataset_name}')
  parser.add_argument('--est_refine_iter', type=int, default=15)
  parser.add_argument('--track_refine_iter', type=int, default=3)
  parser.add_argument('--debug', type=int, default=2)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, zfar=np.inf) # shorter_side=960

  start_frame = 1880 # '0' means the begining
  frame_to_propagate = 1975 # len(reader.color_files)
  from tqdm import tqdm
  progress_bar = tqdm(total=frame_to_propagate - start_frame, desc="Processing frames")
  
  for i in range(start_frame, frame_to_propagate):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    mask = reader.get_mask(i).astype(bool)
    
    #####
    top_left_y, bottom_right_y, top_left_x, bottom_right_x = get_bounding_box_coordinates(mask)
    cropped_mask = mask.copy()[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    cropped_color = color.copy()[top_left_y:bottom_right_y, top_left_x:bottom_right_x,:]
    cropped_color[cropped_mask == 0] = [0,255,0]
    cv2.imwrite('cropped_color_bgr.png',cropped_color)

    cropped_depth = depth.copy()[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("AAAbed", cropped_color.shape)
    KK = reader.K.copy()
    KK[0,2] -= top_left_x
    KK[1,2] -= top_left_y
    #####

    # try:
    update_gain=1.0 if i==start_frame else 0.02
    pose = est.register(update_gain=update_gain, K=KK, rgb=cropped_color, depth=cropped_depth, ob_mask=cropped_mask, iteration=args.est_refine_iter)
      # pose = est.register(update_gain=1, K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    # except:
      # pose = est.track_one(rgb=cropped_color, depth=cropped_depth, K=KK, iteration=args.track_refine_iter)
      # pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.1
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    os.makedirs(f'{reader.video_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{reader.video_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      frame = cv2.cvtColor(cv2.resize(vis[..., ::-1], (1280, 720)), cv2.COLOR_BGR2RGB)
      cv2.imwrite(f'{code_dir}/demo_data/{dataset_name}/rgb_axes_bb.png', frame)


    if debug>=2:
      os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)
      
    progress_bar.update(1)
  
  progress_bar.close()

  # to save the photos as a mp4 file, run `python fp2vid.py`