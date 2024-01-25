import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import argparse


def initialize_camera(config, serial):
    # Set the device serial number
    config.enable_device(serial)
    pipeline = rs.pipeline()
    pipeline.start(config)
    return pipeline


def create_save_folders(base_folder, camera_name):
    color_folder = os.path.join(base_folder, camera_name, "color")
    depth_folder = os.path.join(base_folder, camera_name, "depth")
    os.makedirs(color_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    return color_folder, depth_folder


def save_frames(color_frame, depth_frame, color_folder, depth_folder, frame_id):
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    cv2.imwrite(os.path.join(color_folder, f"{frame_id}.png"), color_image)
    cv2.imwrite(os.path.join(depth_folder, f"{frame_id}.png"), depth_image)


if __name__ == "_main_":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=f"{root_dir}/test_data")
    parser.add_argument("--data_name", type=str, default="exp_0")
    args = parser.parse_args()
    base_folder = os.path.join(args.data_folder, args.data_name)
    os.makedirs(base_folder, exist_ok=True)
    # clean the folder with verification
    print(f"Are you sure to clean the folder {base_folder}? (y/n)")
    answer = input()
    if answer == "y":
        os.system(f"rm -rf {base_folder}/*")
    # Start both pipelines
    mvn_cam_serial = "827112072543"  # Moving Camera
    fix_cam_serial = "021122070657"  # Fixed Camera
    # Setup for camera 1
    mvn_cam_config = rs.config()
    mvn_cam_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    mvn_cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    mvn_cam_pipeline = initialize_camera(mvn_cam_config, mvn_cam_serial)
    mvn_cam_color, mvn_cam_depth = create_save_folders(base_folder, "camera_1")
    mvn_cam_align = rs.align(rs.stream.color)
    # Setup for camera 2
    fix_cam_config = rs.config()
    fix_cam_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    fix_cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    fix_cam_pipeline = initialize_camera(fix_cam_config, fix_cam_serial)
    fix_cam_color, fix_cam_depth = create_save_folders(base_folder, "camera_2")
    fix_cam_align = rs.align(rs.stream.color)
    frame_id = 0
    try:
        while True:
            # Process frames from camera 1
            mvn_cam_frames = mvn_cam_pipeline.wait_for_frames()
            mvn_cam_aligned_frames = mvn_cam_align.process(mvn_cam_frames)
            mvn_cam_color_frame = mvn_cam_aligned_frames.get_color_frame()
            mvn_cam_depth_frame = mvn_cam_aligned_frames.get_depth_frame()
            # Process frames from camera 2
            fix_cam_frames = fix_cam_pipeline.wait_for_frames()
            fix_cam_aligned_frames = fix_cam_align.process(fix_cam_frames)
            fix_cam_color_frame = fix_cam_aligned_frames.get_color_frame()
            fix_cam_depth_frame = fix_cam_aligned_frames.get_depth_frame()
            # Validate that both frames are valid for both cameras
            if not mvn_cam_color_frame or not mvn_cam_depth_frame or not fix_cam_color_frame or not fix_cam_depth_frame:
                continue
            save_frames(mvn_cam_color_frame, mvn_cam_depth_frame, mvn_cam_color, mvn_cam_depth, frame_id)
            save_frames(fix_cam_color_frame, fix_cam_depth_frame, fix_cam_color, fix_cam_depth, frame_id)
            frame_id += 1
            mvn_cam_color_image = np.asanyarray(mvn_cam_color_frame.get_data())
            fix_cam_color_image = np.asanyarray(fix_cam_color_frame.get_data())
            combined_image = np.hstack((mvn_cam_color_image, fix_cam_color_image))
            cv2.namedWindow("Moving(Left) - Fixed (Right)", cv2.WINDOW_NORMAL)
            cv2.imshow("Moving(Left) - Fixed (Right)", combined_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
            # Save camera intrinsics and other information
        for i, pipeline in enumerate([mvn_cam_pipeline, fix_cam_pipeline], start=1):
            profile = pipeline.get_active_profile()
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            depth_intrinsics = depth_profile.get_intrinsics()
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            with open(os.path.join(args.data_folder, args.data_name, f"intrinsics_camera{i}.txt"), "w") as f:
                f.writelines(
                    [
                        f"fx: {depth_intrinsics.fx}\n",
                        f"fy: {depth_intrinsics.fy}\n",
                        f"cx: {depth_intrinsics.ppx}\n",
                        f"cy: {depth_intrinsics.ppy}\n",
                    ]
                )
            cam_info = {
                "intrinsic": [
                    [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                    [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                    [0, 0, 1],
                ],
                "depth_scale": depth_scale,
            }
            json.dump(cam_info, open(os.path.join(args.data_folder, args.data_name, f"cam_info_camera{i}.json"), "w"))
    finally:
        mvn_cam_pipeline.stop()
        fix_cam_pipeline.stop()
