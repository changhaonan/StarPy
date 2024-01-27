"""Call kinectFusion"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import json


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="exp_1")
    parser.add_argument("--num_obj", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--img_format", type=str, default="jpg")
    parser.add_argument("--kf_path", type=str, default="external")
    args = parser.parse_args()

    # Prepare the data folder
    root_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(root_dir, "test_data")
    exp_name = "recon/episode1-2"
    fix_camera_id = 2
    data_dir = os.path.join(raw_data_dir, exp_name)
    seg_dir = os.path.join(data_dir, "seg")
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    # Call kinectFusion
    cam_info_file = os.path.join(data_dir, "cam_info.json")
    with open(cam_info_file) as f:
        cam_info = json.load(f)

    # prepare kf config
    color_image = cv2.imread(os.path.join(data_dir, "color", f"{0}.{args.img_format}"))
    im_w, im_h = color_image.shape[:2]
    kf_config = {
        "id": "0001",
        "im_w": im_w,
        "im_h": im_h,
        "cam_intr": cam_info["intrinsic"],
        "bound_z:": [-0.5, 0.5],
        "depth_scale": 1.0 / cam_info["depth_scale"],
    }
    kf_config_file = os.path.join(data_dir, "config.json")
    with open(kf_config_file, "w") as f:
        json.dump(kf_config, f)

    # Run kinectFusion
    kf_dir = args.kf_path

    os.system(f"cd {kf_dir} && python -m KinectFusion-python.main --dataset {raw_data_dir} --video {exp_name}")
