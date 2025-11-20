import os
import numpy as np
from PIL import Image
from tqdm import tqdm

working_dir = "data/working"
video_dir_list = [os.path.join(working_dir, v) for v in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, v))]
video_dir_list = sorted(video_dir_list)

image_size = (1920, 1080)
image_list = []

for video_dir in video_dir_list:
    frame_dir = os.path.join(video_dir, "frames")
    depth_dir = os.path.join(video_dir, "depth")
    separated_depth_dir = os.path.join(video_dir, "separated_depth")
    enhanced_depth_dir = os.path.join(video_dir, "enhanced_depth")
    smoothed_depth_dir = os.path.join(video_dir, "smoothed_depth")
    
    frame_list = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    depth_map_list = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    separated_depth_map_list = sorted([os.path.join(separated_depth_dir, f) for f in os.listdir(separated_depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    enhanced_depth_map_list = sorted([os.path.join(enhanced_depth_dir, f) for f in os.listdir(enhanced_depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    smoothed_depth_map_list = sorted([os.path.join(smoothed_depth_dir, f) for f in os.listdir(smoothed_depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    first_frame_path = frame_list[0]
    first_depth_map_path = depth_map_list[0]
    first_separated_depth_map_path = separated_depth_map_list[0]
    first_enhanced_depth_map_path = enhanced_depth_map_list[0]
    first_smoothed_depth_map_path = smoothed_depth_map_list[0]
    
    first_frame = Image.open(first_frame_path)
    first_depth_map = Image.open(first_depth_map_path)
    first_separated_depth_map = Image.open(first_separated_depth_map_path)
    first_enhanced_depth_map = Image.open(first_enhanced_depth_map_path)
    first_smoothed_depth_map = Image.open(first_smoothed_depth_map_path)
    
    first_frame = first_frame.resize(image_size)
    first_depth_map = first_depth_map.resize(image_size)
    first_separated_depth_map = first_separated_depth_map.resize(image_size)
    first_enhanced_depth_map = first_enhanced_depth_map.resize(image_size)
    first_smoothed_depth_map = first_smoothed_depth_map.resize(image_size)
    
    image_grid = Image.new('RGB', (first_frame.width * 5, first_frame.height))
    image_grid.paste(first_frame, (0, 0))
    image_grid.paste(first_depth_map.convert('RGB'), (first_frame.width, 0))
    image_grid.paste(first_separated_depth_map.convert('RGB'), (first_frame.width * 2, 0))
    image_grid.paste(first_enhanced_depth_map.convert('RGB'), (first_frame.width * 3, 0))
    image_grid.paste(first_smoothed_depth_map.convert('RGB'), (first_frame.width * 4, 0))
    
    image_list.append(image_grid)
    
output_grid = Image.new('RGB', (image_list[0].width, image_list[0].height * len(image_list)))
for idx, img in enumerate(image_list):
    output_grid.paste(img, (0, idx * img.height))
output_grid.save("output_grid.png")