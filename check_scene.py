"""Check the alignment of the color and depth camera."""
import numpy as np
import open3d as o3d
import os
import json
import argparse
import cv2


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
    parser.add_argument("--exp_name", type=str, default="exp_1")
    parser.add_argument("--num_obj", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--img_format", type=str, default="jpg")
    parser.add_argument("--outlier_method", type=str, default="statistical")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = "test_data"
    exp_name = "recon/episode1-2"
    data_path = os.path.join(root_dir, "test_data", exp_name)
    img_format = args.img_format
    num_obj = args.num_obj
    use_mask = True
    outlier_method = args.outlier_method

    frame = 1200
    # Parse intrinsics
    with open(os.path.join(data_path, "intrinsics.txt")) as f:
        lines = f.readlines()
        fx, fy, cx, cy = [float(line.split(":")[1]) for line in lines]

    # Read cam info
    cam_info_file = os.path.join(data_path, "cam_info.json")
    with open(cam_info_file) as f:
        cam_info = json.load(f)

    color_image = cv2.imread(os.path.join(data_path, "color", f"{frame}.{img_format}"))
    # depth_image = o3d.io.read_image(os.path.join(data_path, "depth", f"{frame}.png"))  # depth is saved as png
    depth_image = (
        cv2.imread(os.path.join(data_path, "depth", f"{frame}.png"), cv2.IMREAD_ANYDEPTH) * cam_info["depth_scale"]
    )
    img_size = np.array(depth_image).shape
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    color_pcd_list = []
    for i in range(num_obj):
        if use_mask:
            seg_image = cv2.imread(os.path.join(data_path, "seg", f"{frame}_{i}.png"), cv2.IMREAD_UNCHANGED)
            # Apply erosion to remove noise
            kernel = np.ones((10, 10), np.uint8)
            seg_image = cv2.erode(seg_image, kernel, iterations=1)
            color_mask_image = color_image * seg_image[:, :, None]
            depth_mask_image = depth_image * seg_image

        # color_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        color_pcd, num_points = get_o3d_pointcloud(color_mask_image, depth_mask_image, intrinsic=intrinsics)

        # Remove outliers
        if outlier_method == "statistical":
            cl, ind = color_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        elif outlier_method == "radius":
            cl, ind = color_pcd.remove_radius_outlier(nb_points=5, radius=1.0)
        else:
            raise NotImplementedError
        color_pcd = color_pcd.select_by_index(ind)
        color_pcd_list.append(color_pcd)

    # Visualize
    o3d.visualization.draw_geometries(color_pcd_list)
