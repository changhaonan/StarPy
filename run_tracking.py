"""Run tracking & perform post-process."""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="exp_1")
    parser.add_argument("--num_obj", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--img_format", type=str, default="jpg")
    parser.add_argument("--pytracking_path", type=str, default="external/pytracking/pytracking")
    args = parser.parse_args()

    # Prepare the data folder
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = "test_data"
    exp_name = "recon/episode1-2"
    fix_camera_id = 2
    data_dir = os.path.join(root_dir, data_dir, exp_name)
    seg_dir = os.path.join(data_dir, "seg")
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    # Call pytracking
    pytracking_dir = args.pytracking_path
    pytracking_result_dir = os.path.join(pytracking_dir, "tracking_results", "rts")
    img_format = args.img_format
    num_obj = args.num_obj
    for idx_obj in range(num_obj):
        print(f"Processing object {idx_obj}...")
        os.system(
            f"cd {pytracking_dir} && python run_video.py rts rts50 {os.path.join(data_dir, f'color/%d.{img_format}')} --save_results"
        )
        # Copy the result from pytracking
        # Read all seg masks
        seg_img_list = os.listdir(pytracking_result_dir)
        seg_img_list = [x for x in seg_img_list if x.endswith(".png") or x.endswith(".jpg")]
        # Sort by numerical order
        seg_img_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

        # Read all images
        for idx, seg_img_file in tqdm(enumerate(seg_img_list)):
            seg_img_file_path = os.path.join(pytracking_result_dir, seg_img_file)
            goal_img_file_path = os.path.join(seg_dir, f"{idx}_{idx_obj}.png")
            # # Copy the image
            # os.system(f"cp {seg_img_file_path} {goal_img_file_path}")
            # Save the foreground as unchar
            seg_img = cv2.imread(seg_img_file_path)
            seg_img = seg_img[:, :, 0]
            seg_img = np.where(seg_img == 0, 0, 255)
            cv2.imwrite(goal_img_file_path, seg_img)
