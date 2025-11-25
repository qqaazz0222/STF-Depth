import os
import pickle
import numpy as np
import PIL.Image as Image

infer_result_path = '/workexternal/hyunsu/SFTDepth/STF-Depth/data/working/vp_test/images_batch/infer_result.pkl'

frame_idx = 0
step_n = 5

with open(infer_result_path, 'rb') as f:
    infer_result = pickle.load(f)
    
frame_list = infer_result['frame']
depth_map_list = infer_result['depth_map']
segment_map_list = infer_result['segment_map']
segment_detail_map_list = infer_result['segment_detail_map']
segment_detail_info_list = infer_result['segment_detail_info']

frame_size = depth_map_list[0].shape[:2]

target_depth_map = depth_map_list[frame_idx]
# Save target_depth_map as grayscale image
depth_normalized = (target_depth_map - np.min(target_depth_map)) / (np.max(target_depth_map) - np.min(target_depth_map)) * 255
depth_normalized = depth_normalized.astype(np.uint8)
depth_img = Image.fromarray(depth_normalized, mode='L')
depth_img.save(f'target_depth_map_frame_{frame_idx}.png')
print(f'Saved target depth map as grayscale image')

print('min depth:', np.min(target_depth_map), 'max depth:', np.max(target_depth_map))
# Create variable step sizes that get smaller as we approach maximum depth
# Using exponential distribution for finer granularity near maximum
depth_min_val = np.min(target_depth_map)
depth_max_val = np.max(target_depth_map)
depth_range = depth_max_val - depth_min_val

# Generate step values with exponential spacing (denser near maximum)
# Using powers to create non-uniform distribution
ratios = np.linspace(0, 1, step_n + 1) ** 2  # Square for exponential curve
step_values = [depth_min_val + depth_range * ratio for ratio in ratios]

center_points = []

print('step values:', step_values)
for i in range(step_n):
    depth_min = step_values[i]
    depth_max = step_values[i + 1]
    
    mask = (target_depth_map >= depth_min) & (target_depth_map < depth_max)
    masked_depth = np.zeros_like(target_depth_map)
    masked_depth[mask] = target_depth_map[mask]

    depth_image = (masked_depth - np.min(masked_depth)) / (np.max(masked_depth) - np.min(masked_depth) + 1e-8) * 255
    depth_image = depth_image.astype(np.uint8)
    
    # Create RGBA image with transparency
    rgba_image = np.zeros((*target_depth_map.shape, 4), dtype=np.uint8)
    rgba_image[mask, :3] = depth_image[mask, np.newaxis]  # RGB channels
    rgba_image[mask, 3] = 255  # Alpha channel (opaque where mask is True)
    
    # Calculate center point of valid values in mask
    valid_coords = np.argwhere(mask)
    if len(valid_coords) > 0:
        center_y, center_x = valid_coords.mean(axis=0).astype(int)
        center_points.append((center_x, center_y))
        print(f'Center point (x, y): ({center_x}, {center_y})')
        # Draw red dot at center point
        # Draw red circle at center point
        radius = 5
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle_mask = x**2 + y**2 <= radius**2
        
        y_min = max(0, center_y - radius)
        y_max = min(rgba_image.shape[0], center_y + radius + 1)
        x_min = max(0, center_x - radius)
        x_max = min(rgba_image.shape[1], center_x + radius + 1)
        
        circle_y_min = radius - (center_y - y_min)
        circle_y_max = radius + (y_max - center_y)
        circle_x_min = radius - (center_x - x_min)
        circle_x_max = radius + (x_max - center_x)
        
        rgba_image[y_min:y_max, x_min:x_max][circle_mask[circle_y_min:circle_y_max, circle_x_min:circle_x_max]] = [255, 0, 0, 255]
    else:
        print('No valid values in this depth range')

    img = Image.fromarray(rgba_image, mode='RGBA')
    img.save(f'depth_range_{i+1}.png')
    print(f'Saved depth range image for range {depth_min:.2f} to {depth_max:.2f}')
    
depthest_point = np.unravel_index(np.argmin(target_depth_map), target_depth_map.shape)
print(f'Deepest point (y, x): {depthest_point}')
center_points.append((depthest_point[1], depthest_point[0]))
    
# Fit a line to the center points using least squares
if len(center_points) >= 2:
    center_points_array = np.array(center_points)
    x_coords = center_points_array[:, 0]
    y_coords = center_points_array[:, 1]
    
    # Fit line: y = mx + b
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    
    print(f'\nFitted line: y = {m:.4f}x + {b:.4f}')
    
    # Create visualization with fitted line
    line_image = np.zeros((*frame_size, 4), dtype=np.uint8)
    
    # Draw the fitted line
    x_line = np.arange(0, frame_size[1])
    y_line = (m * x_line + b).astype(int)
    
    # Filter valid coordinates
    valid_mask = (y_line >= 0) & (y_line < frame_size[0])
    x_line_valid = x_line[valid_mask]
    y_line_valid = y_line[valid_mask]
    
    line_image[y_line_valid, x_line_valid] = [0, 255, 0, 255]  # Green line
    
    # Draw center points
    for cx, cy in center_points:
        radius = 5
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle_mask = x**2 + y**2 <= radius**2
        
        y_min = max(0, cy - radius)
        y_max = min(line_image.shape[0], cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(line_image.shape[1], cx + radius + 1)
        
        circle_y_min = radius - (cy - y_min)
        circle_y_max = radius + (y_max - cy)
        circle_x_min = radius - (cx - x_min)
        circle_x_max = radius + (x_max - cx)
        
        line_image[y_min:y_max, x_min:x_max][circle_mask[circle_y_min:circle_y_max, circle_x_min:circle_x_max]] = [255, 0, 0, 255]  # Red points
    
    img = Image.fromarray(line_image, mode='RGBA')
    img.save('fitted_line_with_centers.png')
    print('Saved fitted line visualization')
else:
    print('Not enough center points to fit a line')