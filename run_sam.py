"""Run sam on tracking masks"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

sam_path = "external/EfficientSAM"
import sys

# Change cwd
# os.chdir(sam_path)
sys.path.append(sam_path)

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from efficient_sam.efficient_sam import build_efficient_sam

# from squeeze_sam.build_squeeze_sam import build_squeeze_sam

from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile


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
    num_obj = args.num_obj
    data_dir = os.path.join(root_dir, data_dir, exp_name)
    seg_dir = os.path.join(data_dir, "sam_seg")
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    # Build the EfficientSAM-Ti model.
    model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=f"{sam_path}/weights/efficient_sam_vitt.pt",
    ).eval().cuda().float()
    # model = build_efficient_sam(
    #     encoder_patch_embed_dim=384,
    #     encoder_num_heads=6,
    #     checkpoint=f"{sam_path}/weights/efficient_sam_vits.pt",
    # ).eval().cuda().float()

    # Compute number of frames
    img_list = os.listdir(os.path.join(data_dir, f"color"))
    img_list = [x for x in img_list if x.endswith(".png") or x.endswith(".jpg")]
    num_frame = len(img_list)

    for idx_obj in range(num_obj):
        # Read bbox
        bbox_file_path = os.path.join(data_dir, f"bbox_{idx_obj}.txt")
        bbox_list = []
        with open(bbox_file_path, "r") as f:
            for line in f:
                bbox_list.append([int(x) for x in line.strip().split("\t")])
        for idx in tqdm(range(num_frame)):
            color_image_np = np.array(Image.open(os.path.join(data_dir, f"color/{idx}.jpg")))
            color_image_tensor = transforms.ToTensor()(color_image_np).cuda().float()
            # BBox
            x0, y0, w, h = bbox_list[idx]
            x1 = x0 + w
            y1 = y0 + h

            input_points = torch.tensor([[[[x0, y0], [x1, y1]]]]).cuda().float()
            input_labels = torch.tensor([[[2, 3]]]).cuda().float()
            predicted_logits, predicted_iou = model(
                color_image_tensor[None, ...],
                input_points,
                input_labels,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
            masked_image_np = color_image_np.copy().astype(np.uint8) * mask[:,:,None]
            Image.fromarray(masked_image_np).save(f"{seg_dir}/{idx}_{idx_obj}_mask.png")