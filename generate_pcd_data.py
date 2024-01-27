"""After getting the camera pose & pcd, we can use this script to generate the pcd data for each object."""
import argparse
import os
import numpy as np
import json
import cv2
import open3d as o3d
import pickle
import tqdm


def get_o3d_pointcloud(color, depth, intrinsic, flip_x: bool = False, flip_y: bool = False):
    """Get Open3D pointcloud from perspective depth image and color image.

    Args:
      color: HxWx3 uint8 array of RGB images.
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    if flip_x:
        px = width - 1 - px
    if flip_y:
        py = height - 1 - py
    px = (px - intrinsic[0, 2]) * (depth / intrinsic[0, 0])
    py = (py - intrinsic[1, 2]) * (depth / intrinsic[1, 1])
    # Stack the coordinates and reshape
    points = np.float32([px, py, depth]).transpose(1, 2, 0).reshape(-1, 3)

    # Assuming color image is in the format height x width x 3 (RGB)
    # Reshape color image to align with points
    colors = color.reshape(-1, 3)

    pcolors = np.hstack((points, colors))
    pcolors = pcolors[pcolors[:, 0] != 0.0, :]
    if pcolors.shape[0] == 0:
        return None, 0

    points = pcolors[:, :3]
    colors = pcolors[:, 3:]

    tpcd = o3d.geometry.PointCloud()
    tpcd.points = o3d.utility.Vector3dVector(points)
    tpcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Estimate normals
    tpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # Optional: Orient the normals to be consistent
    tpcd.orient_normals_consistent_tangent_plane(k=50)

    # return pcd_with_color
    return tpcd, points.shape[0]


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_obj", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--img_format", type=str, default="jpg")
    parser.add_argument("--outlier_method", type=str, default="statistical")
    parser.add_argument("--families", type=str, default="tag16h5")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    img_format = args.img_format
    num_obj = args.num_obj
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "test_data", "recon")
    outlier_method = args.outlier_method
    use_mask = True

    experiment_list = ["episode1-0", "episode1-1", "episode1-2"]
    pcd_size = 2048

    # Prepare the data
    pcd_dict = {}
    semantic_dict = {}
    # Read the camera pose
    for exp_name in tqdm.tqdm(experiment_list):
        exp_data_dir = os.path.join(data_dir, exp_name)
        # Read the camera pose
        kf_results_file = os.path.join(exp_data_dir, "kf_results.npz")
        kf_results = np.load(kf_results_file)
        cam_poses = kf_results["cam_poses"]

        # Read semantics
        semantic_file = os.path.join(exp_data_dir, "semantics.txt")
        with open(semantic_file) as f:
            lines = f.readlines()
            semantic_list = [line.strip() for line in lines]
            for semantic in semantic_list:
                if semantic not in semantic_dict:
                    semantic_dict[semantic] = len(semantic_dict)

        num_frame = cam_poses.shape[0]
        for frame in tqdm.tqdm(range(num_frame), leave=False):
            cam_pose = cam_poses[frame]
            # Build pcd
            # Parse intrinsics
            with open(os.path.join(exp_data_dir, "intrinsics.txt")) as f:
                lines = f.readlines()
                fx, fy, cx, cy = [float(line.split(":")[1]) for line in lines]

            # Read cam info
            cam_info_file = os.path.join(exp_data_dir, "cam_info.json")
            with open(cam_info_file) as f:
                cam_info = json.load(f)

            color_image = cv2.imread(os.path.join(exp_data_dir, "color", f"{frame}.{img_format}"))
            # depth_image = o3d.io.read_image(os.path.join(exp_data_dir, "depth", f"{frame}.png"))  # depth is saved as png
            depth_image = (
                cv2.imread(os.path.join(exp_data_dir, "depth", f"{frame}.png"), cv2.IMREAD_ANYDEPTH)
                * cam_info["depth_scale"]
            )
            img_size = np.array(depth_image).shape
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            color_pcd_list = []
            for i in range(num_obj):
                if use_mask:
                    seg_image = cv2.imread(os.path.join(exp_data_dir, "seg", f"{frame}_{i}.png"), cv2.IMREAD_UNCHANGED)
                    seg_image = np.where(seg_image == 0, 0, 1)
                    color_mask_image = color_image * seg_image[:, :, None]
                    depth_mask_image = depth_image * seg_image

                # color_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
                color_pcd, num_points = get_o3d_pointcloud(color_mask_image, depth_mask_image, intrinsic=intrinsics)

                # Remove outliers
                if outlier_method == "statistical":
                    cl, ind = color_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
                elif outlier_method == "radius":
                    cl, ind = color_pcd.remove_radius_outlier(nb_points=5, radius=1.0)
                else:
                    raise NotImplementedError
                color_pcd = color_pcd.select_by_index(ind)
                # Downsample to pcd_size
                if len(ind) > pcd_size:
                    color_pcd = color_pcd.farthest_point_down_sample(pcd_size)

                # Transform the pcd
                color_pcd.transform(cam_pose)
                color_pcd_list.append(color_pcd)

                # Log pcd array
                obj_name = f"object_{i}"
                if obj_name not in pcd_dict:
                    pcd_dict[obj_name] = []
                # Convert to numpy array
                pcd_pos = np.asarray(color_pcd.points)
                pcd_normal = np.asarray(color_pcd.normals)
                pcd_color = np.asarray(color_pcd.colors)
                pcd_array = np.concatenate([pcd_pos, pcd_normal, pcd_color], axis=1)
                pcd_dict[obj_name].append(pcd_array)

                # Log semantic array
                semantic_name = semantic_list[i]
                semantic_id = semantic_dict[semantic_name]
                obj_semantic_name = f"object_{i}_semantic"
                if obj_semantic_name not in pcd_dict:
                    pcd_dict[obj_semantic_name] = []
                pcd_dict[obj_semantic_name].append(semantic_id)

            # Visualize
            # origin = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
            # o3d.visualization.draw_geometries(color_pcd_list + [origin])

        # Save the pcd dict
        export_path = os.path.join(data_dir, f"exp_{exp_name}.pkl")
        with open(export_path, "wb") as f:
            pickle.dump(pcd_dict, f)
