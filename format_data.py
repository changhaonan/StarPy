"""Format data for reconstruction and tracking."""
import os
import cv2
import numpy as np


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = "test_data"
    exp_name = "exp_0"
    fix_camera_id = 2
    data_dir = os.path.join(data_dir, exp_name)
    seg_dir = os.path.join(data_dir, f"camera_{fix_camera_id}", "seg")
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    # Copy the result from pytracking
    pytracking_dir = os.path.join(root_dir, "external", "pytracking", "pytracking")
    pytracking_result_dir = os.path.join(pytracking_dir, "tracking_results", "rts")

    # Read all seg masks
    seg_img_list = os.listdir(pytracking_result_dir)
    seg_img_list = [x for x in seg_img_list if x.endswith(".png") or x.endswith(".jpg")]
    # Sort by numerical order
    seg_img_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

    # Read all images
    for idx, seg_img_file in enumerate(seg_img_list):
        seg_img_file_path = os.path.join(pytracking_result_dir, seg_img_file)
        goal_img_file_path = os.path.join(seg_dir, f"{idx}.png")
        # # Copy the image
        # os.system(f"cp {seg_img_file_path} {goal_img_file_path}")
        # Save the foreground as unchar
        seg_img = cv2.imread(seg_img_file_path)
        seg_img = seg_img[:, :, 0]
        seg_img = np.where(seg_img == 0, 0, 255)
        cv2.imwrite(goal_img_file_path, seg_img)
