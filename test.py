# 

import os
import csv
import time
import rich
import pickle
from tqdm import tqdm
from functools import wraps
from argparse import ArgumentParser
from collections import defaultdict
import warnings
import logging
from multiprocessing import Pool, cpu_count

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from sklearn.linear_model import RANSACRegressor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers.models.oneformer.image_processing_oneformer").setLevel(logging.ERROR)


def print_task(func):
    """작업 시작과 종료를 알리는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name_upper = func.__name__.upper()
        func_name_upper = func_name_upper.replace('_', ' ')
        func_name_upper = func_name_upper.replace('INIT ', 'INITIALIZING ')
        rich.print(f"[bold green][+] {func_name_upper}...[/bold green]")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        rich.print(f"[bold green][=] {func_name_upper} Done![/bold green] [gray]{elapsed_time:.2f}s\n{'-'*50}[/gray]")
        return result
    return wrapper

@print_task
def init_args():
    """명령줄 인자 파서 초기화 함수"""
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./test/data/input", help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default="./test/data/output", help='Output directory to save results')
    parser.add_argument('--working_dir', type=str, default="./test/data/working", help='Working directory for intermediate files')
    parser.add_argument('--datasets', type=list, default=["nyu", "kitti"], help='Dataset names for depth estimation')
    parser.add_argument('--depth_model', type=list, default=["intel-isl/MiDaS", "DPT_Large"], help='Depth estimation model name')
    parser.add_argument('--depth_utils', type=list, default=["intel-isl/MiDaS", "transforms"], help='Depth estimation utilities name')
    parser.add_argument('--segment_model', type=list, default=['pytorch/vision:v0.10.0', 'deeplabv3_resnet101'], help='Segmentation model name')
    parser.add_argument('--segment_detail_model', type=list, default=['shi-labs/oneformer_coco_swin_large'], help='Segmentation model name')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of layers for processing')
    parser.add_argument('--infer_result', type=str, default="infer_result.pkl", help='Inference result filename')
    parser.add_argument('--separated_result', type=str, default="separated_result.pkl", help='Separated result filename')
    parser.add_argument('--enhanced_result', type=str, default="enhanced_result.pkl", help='Enhanced result filename')
    parser.add_argument('--smoothed_result', type=str, default="smoothed_result.pkl", help='Smoothed result filename')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the results')
    
    args = parser.parse_args()
    data_dir = os.path.dirname(args.input_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    return args

@print_task
def init_model(args):
    """모델 초기화 함수"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rich.print(f" - Using device: {device}")
    
    # 깊이 추정 모델 초기화
    rich.print(" - Initializing depth model...")
    depth_model = torch.hub.load(args.depth_model[0], args.depth_model[1], pretrained=True)
    depth_model.to(device)
    depth_model.eval()
    rich.print(" - Depth model initialized.")
    
    # 세그멘테이션 모델 초기화
    rich.print(" - Initializing segmentation model...")
    segment_model = torch.hub.load(args.segment_model[0], args.segment_model[1], pretrained=True)
    segment_model.to(device)
    segment_model.eval()
    rich.print(" - Segmentation model initialized.")
    
    # 세그멘테이션 상세 모델 초기화
    rich.print(" - Initializing segmentation detail model...")
    segment_detail_model = AutoModelForUniversalSegmentation.from_pretrained(args.segment_detail_model[0])
    segment_detail_model.to(device)
    segment_detail_model.eval()
    rich.print(" - Segmentation detail model initialized.")

    return depth_model, segment_model, segment_detail_model, device


@print_task
def init_processor(args):
    """프로세서 초기화 함수"""
    # 깊이 추정 모델용 프로세서 초기화
    rich.print(" - Initializing depth processor...")
    depth_processor = torch.hub.load(args.depth_utils[0], args.depth_utils[1])
    depth_processor = depth_processor.dpt_transform if args.depth_model[1] == "DPT_Large" else depth_processor.small_transform
    rich.print(" - Depth processor initialized.")

    # 세그멘테이션 모델용 프로세서 초기화
    rich.print(" - Initializing segmentation processor...")
    segment_processor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rich.print(" - Segmentation processor initialized.")

    # 세그멘테이션 상세 모델용 프로세서 초기화
    rich.print(" - Initializing segmentation detail processor...")
    segment_detail_processor = AutoProcessor.from_pretrained(args.segment_detail_model[0])
    rich.print(" - Segmentation detail processor initialized.")
    return depth_processor, segment_processor, segment_detail_processor


@print_task
def load_data(args, dataset_name):
    """데이터 로드 함수"""
    def extract_video_to_frames(video_path, working_dir):
        """비디오에서 프레임 추출 함수"""
        frame_list = []
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            frame_path = os.path.join(working_dir, f"{count:08d}.jpg")
            frame_list.append(frame_path)
            cv2.imwrite(frame_path, image)
            success, image = vidcap.read()
            count += 1
        vidcap.release()
        return frame_list
    
    input_dir = os.path.join(args.input_dir, dataset_name, "input")
    working_dir = os.path.join(args.working_dir, dataset_name)
    os.makedirs(working_dir, exist_ok=True)
    
    image_ext_list = ['.jpg', '.png']
    video_ext_list = ['.mp4', '.avi', '.mov']
    
    dataset = []
    
    # 이미지 파일들을 하나의 배치로 묶기
    image_list = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.lower().endswith(tuple(image_ext_list))]
    image_list.sort()
    
    if len(image_list) > 0:
        # 모든 이미지를 하나의 데이터로 묶어서 처리
        image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in image_list]
        dataset.append({
            'name': 'images_batch',  # 전체 이미지 배치의 이름
            'path': input_dir,
            'frames': image_list,
            'frame_names': image_names,  # 각 이미지의 원본 이름 저장
            'type': 'images'
        })
    
    # 비디오 파일들은 개별적으로 처리
    video_list = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.lower().endswith(tuple(video_ext_list))]
    video_list.sort()
    for video_path in tqdm(video_list, desc=" - Loading videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cur_video_working_dir = os.path.join(working_dir, video_name)
        cur_video_frame_dir = os.path.join(cur_video_working_dir, "frames")
        os.makedirs(cur_video_working_dir, exist_ok=True)
        os.makedirs(cur_video_frame_dir, exist_ok=True)
        frame_list = extract_video_to_frames(video_path, cur_video_frame_dir)
        dataset.append({'name': video_name, 'path': video_path, 'frames': frame_list, 'type': 'video'})
        
    return dataset


@print_task
def load_calc_data(args, dataset_name, stage_name='smoothed'):
    """계산용 데이터 로드 함수"""
    dataset = []
    t_flag, f_flag = 0, 0
    
    pred_dir = os.path.join(args.output_dir, dataset_name)
    if stage_name is not None:
        stage_dir = os.path.join(pred_dir, 'stages', stage_name)
        if os.path.exists(stage_dir):
            pred_dir = stage_dir
        else:
            print(f" - Stage '{stage_name}' outputs not found for {dataset_name}. Skipping load.")
            return []

    pred_file_list = []
    for root, _, files in os.walk(pred_dir):
        if 'depth_map.npy' in files:
            pred_file_list.append(os.path.join(root, 'depth_map.npy'))
    pred_file_list.sort()
    print(f" - Prediction files loaded: {len(pred_file_list)} files found.")
    
    gt_dir = os.path.join(args.input_dir, dataset_name, "gt")
    gt_file_list = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.lower().endswith('.npy')]
    gt_file_list.sort()
    print(f" - Ground Truth files loaded: {len(gt_file_list)} files found.")
    
    for pred, gt in zip(pred_file_list, gt_file_list):
        flag = os.path.basename(gt).split('.')[0] in pred
        if flag:
            t_flag += 1
            dataset.append({
                'pred_path': pred, 
                'pred': np.load(pred),
                'gt_path': gt,
                'gt': np.load(gt)
            })
        else:
            f_flag += 1
    
    print(f" - Pred-GT matching results: Matched: {t_flag}, Unmatched: {f_flag}")
    
    return dataset

def infer_depth(model, processor, image, device):
    """깊이 추정 함수"""
    image_numpy = np.array(image)
    input_batch = processor(image_numpy).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)
        output = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    output_numpy = output.cpu().numpy()
    return output_numpy


def estimate_offset_equation(depth_map, vp):
    """소실점 기반 Offset 방정식 계산 함수"""
    min_depth_value = np.min(depth_map)
    max_depth_value = np.max(depth_map)
    depth_gap = max_depth_value - min_depth_value
    
    if depth_gap == 0:
        return 0.0, 0.0
        
    if vp is not None:
        slope = 0.1 * depth_gap / max_depth_value if max_depth_value != 0 else 0.0
    else:
        slope = 0.05 * depth_gap / max_depth_value if max_depth_value != 0 else 0.0
        
    # min_depth에서 offset = 0이 되도록 intercept 계산
    intercept = -slope * min_depth_value
        
    return slope, intercept


def infer_segment(model, processor, image, device):
    """세그멘테이션 함수(요소와 배경 구분)"""
    input_tensor = processor(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
        
    output_predictions = output.argmax(0)
    return output_predictions


def infer_segment_detail(model, processor, image, device):
    """세그멘테이션 상세 함수(객체별 분할)"""
    inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    output_panoptic_map = result['segmentation']
    output_segments = result['segments_info']
    
    return output_panoptic_map, output_segments


def draw_depth_map(path, depth_map):
    """깊이 맵 시각화 함수 (최적화)"""
    normalized_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    colored_map = (plt.cm.magma_r(normalized_map)[:, :, :3] * 255).astype(np.uint8)
    if path is not None:
        Image.fromarray(colored_map).save(path)
    return colored_map


def draw_segment_map(path, image, seg_map):
    """세그멘테이션 맵 시각화 함수 (최적화)"""
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    segment_map_image = Image.fromarray(seg_map.byte().cpu().numpy()).resize(image.size)
    segment_map_image.putpalette(colors)
    segment_map_image = segment_map_image.convert("RGBA")
    blended_image = Image.blend(image.convert("RGBA"), segment_map_image, alpha=0.7)
    blended_image.convert("RGB").save(path)

    return blended_image


def draw_panoptic_map(path, image, panoptic_map, segments_info):
    """Panoptic 분할 맵 시각화 함수 (최적화)"""
    color_palette = defaultdict(lambda: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    segmented_image_np = np.zeros((panoptic_map.shape[0], panoptic_map.shape[1], 3), dtype=np.uint8)
    panoptic_map_np = panoptic_map.cpu().numpy()

    for segment in segments_info:
        mask = (panoptic_map_np == segment['id'])
        segmented_image_np[mask] = color_palette[segment['id']]

    panoptic_map_image = Image.fromarray(segmented_image_np).convert("RGBA")
    blended_image = Image.blend(image.convert("RGBA"), panoptic_map_image, alpha=0.6)
    blended_image.convert("RGB").save(path)
    
    return blended_image


def save_stage_outputs(dataset_output_dir, video_name, stage_name, depth_map_list, data_type, frame_names=None):
    """단계별 평가를 위한 중간 결과 저장"""
    if depth_map_list is None or len(depth_map_list) == 0:
        return
    stage_video_dir = os.path.join(dataset_output_dir, "stages", stage_name, video_name)
    os.makedirs(stage_video_dir, exist_ok=True)

    if data_type == 'video':
        depth_stack = np.array(depth_map_list, dtype=np.float32)
        np.save(os.path.join(stage_video_dir, "depth_map.npy"), depth_stack)
    else:
        if frame_names is None:
            frame_names = [f"{idx:08d}" for idx in range(len(depth_map_list))]
        for depth_map, frame_name in zip(depth_map_list, frame_names):
            frame_dir = os.path.join(stage_video_dir, frame_name)
            os.makedirs(frame_dir, exist_ok=True)
            np.save(os.path.join(frame_dir, "depth_map.npy"), depth_map)


@print_task
def visualize_results(working_dir, frame_list, depth_map_list, segment_map_list, segment_detail_map_list, segment_detail_info_list):
    """결과 시각화 함수"""
    # 디렉토리 생성
    depth_map_dir = os.path.join(working_dir, "depth")
    segment_map_dir = os.path.join(working_dir, "segment")
    segment_detail_map_dir = os.path.join(working_dir, "segment_detail")
    os.makedirs(depth_map_dir, exist_ok=True)
    os.makedirs(segment_map_dir, exist_ok=True)
    os.makedirs(segment_detail_map_dir, exist_ok=True)

    for idx, depth_map in tqdm(enumerate(depth_map_list), desc=f" - Visualizing depth map", total=len(frame_list)):
        depth_map_path = os.path.join(depth_map_dir, f"{idx:08d}.png")
        draw_depth_map(depth_map_path, depth_map)

    # for idx, (image, segment_map) in tqdm(enumerate(zip(frame_list, segment_map_list)), desc=f" - Visualizing segment map", total=len(frame_list)):
    #     segment_map_path = os.path.join(segment_map_dir, f"{idx:08d}.png")
    #     draw_segment_map(segment_map_path, image, segment_map)
        
    # for idx, (image, segment_detail_map, segments_info) in tqdm(enumerate(zip(frame_list, segment_detail_map_list, segment_detail_info_list)), desc=f" - Visualizing segment detail map", total=len(frame_list)):
    #     segment_detail_map_path = os.path.join(segment_detail_map_dir, f"{idx:08d}.png")
    #     draw_panoptic_map(segment_detail_map_path, image, segment_detail_map, segments_info)
    
    return None


def _estimate_vanishing_point_single(args):
    """깊이맵에서 소실점 추정 함수 (multiprocessing용)"""
    depth_map, num_layers = args
    cur_vanishing_point_list = []
    
    # 깊이 맵 수정 (원본 값이 클수록 가까운 것으로 가정 -> 작은 값이 가까운 것으로 변환)
    depth_map = np.max(depth_map) - depth_map
    
    # 깊이 맵의 최소 및 최대 깊이 값 계산
    min_depth_value = np.min(depth_map) # 가장 가까운 깊이
    max_depth_value = np.max(depth_map) # 가장 먼 깊이
    depth_gap = max_depth_value - min_depth_value
    depth_step = depth_gap / num_layers
    
    # 레이어별 소실점 계산
    for layer_idx in range(num_layers):
        layer_depth_min = min_depth_value + layer_idx * depth_step
        layer_depth_max = min_depth_value + (layer_idx + 1) * depth_step
        binary_mask = np.logical_and(depth_map >= layer_depth_min, depth_map < layer_depth_max).astype(np.uint8)
        M = cv2.moments(binary_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            vanishing_point = (cX, cY)
        else:
            vanishing_point = None
        cur_vanishing_point_list.append(vanishing_point)
        
    # 가장 깊은 점 추가
    deepest_point = np.unravel_index(np.argmax(depth_map, axis=None), depth_map.shape)
    cur_vanishing_point_list.append((int(deepest_point[1]), int(deepest_point[0])))
    
    # 프레임의 최종 소실점 계산: 점들을 가장 잘 반영하는 선 찾기 (RANSAC)
    points = np.array([pt for pt in cur_vanishing_point_list if pt is not None])
    if len(points) > 1:            
        # RANSAC을 이용한 선형 회귀
        ransac = RANSACRegressor()
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        ransac.fit(X, y)
        line_slope = ransac.estimator_.coef_[0]
        line_intercept = ransac.estimator_.intercept_
        
        # 가장 깊은 점 주변 영역에서 선 위의 점 찾기
        deepest_x, deepest_y = deepest_point[1], deepest_point[0]
        search_radius = 50  # 탐색 반경
        min_dist = float('inf')
        final_vanishing_point = None
        
        for x in range(deepest_x - search_radius, deepest_x + search_radius):
            y = int(line_slope * x + line_intercept)
            dist = np.sqrt((x - deepest_x)**2 + (y - deepest_y)**2)
            if dist < min_dist:
                min_dist = dist
                final_vanishing_point = (x, y)
    else:
        final_vanishing_point = cur_vanishing_point_list[-1]
        
    return final_vanishing_point


@print_task
def estimate_vanishing_point(depth_map_list, num_layers=5):
    """소실점 추정 함수"""
    # CPU 코어 수만큼 병렬 처리
    num_cores = cpu_count()
    with Pool(num_cores) as pool:
        vanishing_point_list = list(tqdm(
            pool.imap(_estimate_vanishing_point_single, [(depth_map, num_layers) for depth_map in depth_map_list]),
            total=len(depth_map_list),
            desc=" - Estimating vanishing points"
        ))
    
    return vanishing_point_list


def _separate_process_frame(args):
    """배경과 전경 분리 프레임 처리 (multiprocessing용)"""
    idx, depth_map, segment_map, vp, separated_dir = args
    separated_depth_map = depth_map.copy()
    h, w = separated_depth_map.shape
    binary_mask = (segment_map != 0).astype(np.float32)  # 배경(0)과 요소(1) 구분
    mask_uint8 = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bottom_point_list = []
    
    for contour in contours:
        if len(contour) > 0:
            bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
            bottom_point_list.append(bottom_point)
            
    if bottom_point_list: 
        slope, intercept = estimate_offset_equation(separated_depth_map, vp)
        # 벡터화된 연산으로 픽셀 단위 루프 제거
        mask_indices = binary_mask == 1
        offset = slope * separated_depth_map + intercept
        separated_depth_map[mask_indices] += offset[mask_indices]  # 배경 깊이 조정
    
    separated_depth_map_path = os.path.join(separated_dir, f"{idx:08d}.png")
    draw_depth_map(separated_depth_map_path, separated_depth_map)
    
    return separated_depth_map


@print_task
def separate(working_dir, depth_map_list, segment_map_list, vanishing_point_list):
    """배경과 전경의 구분을 강화하는 함수"""
    separated_dir = os.path.join(working_dir, "separated_depth")
    os.makedirs(separated_dir, exist_ok=True)
    
    # segment_map을 numpy 배열로 변환 (torch tensor인 경우)
    segment_map_numpy_list = []
    for seg_map in segment_map_list:
        if hasattr(seg_map, 'cpu'):
            segment_map_numpy_list.append(seg_map.cpu().numpy())
        else:
            segment_map_numpy_list.append(seg_map)
    
    # CPU 코어 수만큼 병렬 처리
    num_cores = cpu_count()
    args_list = [(idx, depth_map, segment_map, vp, separated_dir) 
                 for idx, (depth_map, segment_map, vp) in enumerate(zip(depth_map_list, segment_map_numpy_list, vanishing_point_list))]
    
    with Pool(num_cores) as pool:
        separated_depth_map_list = list(tqdm(
            pool.imap(_separate_process_frame, args_list),
            total=len(args_list),
            desc=" - Separating depth map"
        ))
    
    return separated_depth_map_list

def _enhance_process_frame(args):
    """객체별 깊이 향상 프레임 처리 (multiprocessing용)"""
    idx, separated_depth_map, segment_detail_map, segment_detail_info, vp, enhanced_dir, label_map, stuff_instance_start_idx, instance_threshold = args
    enhanced_depth_map = separated_depth_map.copy()
    h, w = enhanced_depth_map.shape
    panoptic_map_np = segment_detail_map
    
    info = {x['id']: x for x in segment_detail_info}
    
    # 전역 offset equation 계산
    global_slope, global_intercept = estimate_offset_equation(enhanced_depth_map, vp)
    global_max_distance_to_vp = 0
    if vp[0] > w / 2:
        if vp[1] < h / 2:
            global_max_distance_to_vp = np.sqrt((0 - vp[0])**2 + (0 - vp[1])**2)
        else:
            global_max_distance_to_vp = np.sqrt((0 - vp[0])**2 + (h - vp[1])**2)
    else:
        if vp[1] < h / 2:
            global_max_distance_to_vp = np.sqrt((w - vp[0])**2 + (0 - vp[1])**2)
        else:
            global_max_distance_to_vp = np.sqrt((w - vp[0])**2 + (h - vp[1])**2)
    
    # 인스턴스별 데이터 수집 (벡터화 및 최적화)
    unique_ids = np.unique(panoptic_map_np)
    instance_data = {}
    
    for segment_id in unique_ids:
        mask = (panoptic_map_np == segment_id)
        positions = np.argwhere(mask)
        
        if len(positions) == 0:
            continue
            
        instance_data[segment_id] = {
            'class_id': info[segment_id]['label_id'] if segment_id in info else -1,
            'class_name': label_map.get(str(info[segment_id]['label_id']), "unknown") if segment_id in info else "unknown",
            'score': info[segment_id]['score'] if segment_id in info else 0.0,
            'depths': enhanced_depth_map[mask].tolist(),
            'positions': [(int(y), int(x)) for y, x in positions],
            'min_x': int(positions[:, 1].min()),
            'max_x': int(positions[:, 1].max()),
            'min_y': int(positions[:, 0].min()),
            'max_y': int(positions[:, 0].max())
        }
    
    # 인스턴스별 평균 깊이 계산 및 정렬
    instance_list = []
    for seg_id, data in instance_data.items():
        mean_depth = float(np.mean(data['depths']))
        instance_list.append({
            'id': int(seg_id),
            'class_id': data['class_id'],
            'class': data['class_name'],
            'is_stuff': data['class_id'] >= stuff_instance_start_idx,
            'score': data['score'],
            'mean_depth': mean_depth,
            'min_x': data['min_x'],
            'max_x': data['max_x'],
            'min_y': data['min_y'],
            'max_y': data['max_y'],
            'positions': data['positions']
        })
    
    # 평균 깊이 기준으로 정렬 (깊이가 큰 것이 먼저 = 멀리 있는 것)
    instance_list.sort(key=lambda item: item['mean_depth'], reverse=True)
    
    # 겹침 감지 및 분리 강화 (벡터화)
    for i, instance_i in enumerate(instance_list):
        for j in range(i + 1, len(instance_list)):
            instance_j = instance_list[j]
            
            # y 좌표가 겹치는지 확인
            if not (instance_i['max_y'] < instance_j['min_y'] or instance_j['max_y'] < instance_i['min_y']):
                # 겹침이 발견됨 - instance_j가 더 가까우므로 깊이 차이 강화
                depth_diff = instance_j['mean_depth'] - instance_i['mean_depth']
                if depth_diff > 0 and len(instance_j['positions']) > 0:
                    # 겹친 영역의 깊이 차이를 더 명확하게 (벡터화)
                    overlap_factor = 1.5
                    y_coords, x_coords = zip(*instance_j['positions'])
                    enhanced_depth_map[y_coords, x_coords] += depth_diff * overlap_factor
    
    # 인스턴스별 offset 적용 (벡터화)
    for instance in instance_list:
        seg_id = instance['id']
        seg_class = instance['class']
        seg_is_stuff = instance['is_stuff']
        seg_score = instance['score']
        mean_depth = instance['mean_depth']
        
        np_pos = np.array(instance['positions'])
        center_x = (np_pos[:, 1].min() + np_pos[:, 1].max()) / 2
        center_y = (np_pos[:, 0].min() + np_pos[:, 0].max()) / 2
        distance_to_vp = np.sqrt((center_x - vp[0])**2 + (center_y - vp[1])**2)
        
        # 해당 인스턴스의 평균 깊이를 기반으로 개별 offset 계산
        instance_offset = (global_slope * mean_depth + global_intercept)
        
        # 소실점 기반 기반 추가 offset (가까운 객체일수록 더 큰 보정)
        vp_based_offset = (distance_to_vp / (global_max_distance_to_vp * 0.7)) * 0.1 # TODO 조정 가능
        
        final_offset = instance_offset + vp_based_offset
        if seg_is_stuff:
            final_offset *= -1
        if seg_score < instance_threshold:
            final_offset = 0
        
        # 벡터화된 연산으로 인스턴스의 모든 픽셀에 offset 적용
        if len(instance['positions']) > 0:
            y_coords, x_coords = zip(*instance['positions'])
            enhanced_depth_map[y_coords, x_coords] += final_offset
                
    enhanced_depth_map_path = os.path.join(enhanced_dir, f"{idx:08d}.png")
    draw_depth_map(enhanced_depth_map_path, enhanced_depth_map)
    
    return enhanced_depth_map


@print_task
def enhance(working_dir, separated_depth_map_list, segment_detail_map_list, segment_detail_info_list, vanishing_point_list):
    """객체별 깊이 구분을 강화하는 함수"""
    label_map = {
        "0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane",
        "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light",
        "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench",
        "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep",
        "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe",
        "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase",
        "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite",
        "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard",
        "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork",
        "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple",
        "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog",
        "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch",
        "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv",
        "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone",
        "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator",
        "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear",
        "78": "hair drier", "79": "toothbrush", "80": "banner", "81": "blanket", "82": "branch",
        "83": "bridge", "84": "building-other", "85": "bush", "86": "cabinet", "87": "cage",
        "88": "cardboard", "89": "carpet", "90": "ceiling-other", "91": "ceiling-tile", "92": "cloth",
        "93": "clothes", "94": "clouds", "95": "counter", "96": "cupboard", "97": "curtain",
        "98": "desk-stuff", "99": "dirt", "100": "door-stuff", "101": "fence", "102": "floor-marble",
        "103": "floor-other", "104": "floor-stone", "105": "floor-tile", "106": "floor-wood", "107": "flower",
        "108": "fog", "109": "food-other", "110": "fruit", "111": "furniture-other", "112": "grass",
        "113": "gravel", "114": "ground-other", "115": "hill", "116": "house", "117": "leaves",
        "118": "light", "119": "mat", "120": "metal", "121": "mirror-stuff", "122": "moss",
        "123": "mountain", "124": "mud", "125": "napkin", "126": "net", "127": "paper",
        "128": "pavement", "129": "pillow", "130": "plant-other", "131": "plastic", "132": "platform"
    }
    stuff_instance_start_idx = 80  # stuff 클래스의 시작 인덱스
    instance_threshold = 0.7
    
    enhanced_dir = os.path.join(working_dir, "enhanced_depth")
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # segment_detail_map을 numpy 배열로 변환 (torch tensor인 경우)
    segment_detail_map_numpy_list = []
    for seg_detail_map in segment_detail_map_list:
        if hasattr(seg_detail_map, 'cpu'):
            segment_detail_map_numpy_list.append(seg_detail_map.cpu().numpy())
        else:
            segment_detail_map_numpy_list.append(seg_detail_map)
    
    # CPU 코어 수만큼 병렬 처리
    num_cores = cpu_count()
    args_list = [(idx, separated_depth_map, segment_detail_map, segment_detail_info, vp, enhanced_dir, label_map, stuff_instance_start_idx, instance_threshold)
                 for idx, (separated_depth_map, segment_detail_map, segment_detail_info, vp) in 
                 enumerate(zip(separated_depth_map_list, segment_detail_map_numpy_list, segment_detail_info_list, vanishing_point_list))]
    
    with Pool(num_cores) as pool:
        enhanced_depth_map_list = list(tqdm(
            pool.imap(_enhance_process_frame, args_list),
            total=len(args_list),
            desc=" - Enhancing depth map"
        ))

    return enhanced_depth_map_list


def _temporal_smooth_frame(args):
    """시간적 일관성 프레임 스무딩 (multiprocessing용 - 최적화)"""
    i, num_frames, lambda_val, prev_map, curr_map, next_map = args
    
    if i == 0:
        new_map = (1 - lambda_val) * curr_map + lambda_val * next_map
    elif i == num_frames - 1:
        new_map = (1 - lambda_val) * curr_map + lambda_val * prev_map
    else:
        new_map = (1 - 2 * lambda_val) * curr_map + lambda_val * (prev_map + next_map)
    
    return new_map


@print_task
def temporal_smooth(video_working_dir, enhanced_depth_map_list, lambda_val=0.5, num_iterations=2):
    """시간적 일관성을 위한 깊이 맵 스무딩 함수 (최적화)"""
    smoothed_dir = os.path.join(video_working_dir, "smoothed_depth")
    os.makedirs(smoothed_dir, exist_ok=True)
    num_frames = len(enhanced_depth_map_list)
    
    # NumPy 배열로 변환 (한 번만)
    smoothed_depth_maps = np.array(enhanced_depth_map_list, dtype=np.float32)
    
    if num_frames > 3:
        num_cores = min(cpu_count(), 16)  # 너무 많은 프로세스 방지
        
        for iteration in range(num_iterations):
            # 병렬 처리를 위한 인자 준비 (전체 배열 대신 필요한 프레임만 전달)
            args_list = []
            for i in range(num_frames):
                prev_map = smoothed_depth_maps[i-1] if i > 0 else None
                curr_map = smoothed_depth_maps[i]
                next_map = smoothed_depth_maps[i+1] if i < num_frames - 1 else None
                args_list.append((i, num_frames, lambda_val, prev_map, curr_map, next_map))
            
            with Pool(num_cores) as pool:
                new_smoothed_list = list(tqdm(
                    pool.imap(_temporal_smooth_frame, args_list),
                    total=num_frames,
                    desc=f" - Smoothing depth map, iteration {iteration + 1}/{num_iterations}"
                ))
            
            # 결과를 배열로 업데이트
            smoothed_depth_maps = np.array(new_smoothed_list, dtype=np.float32)
    
    # 결과 저장 (병렬화 가능)
    for i in range(num_frames):
        smoothed_depth_map_path = os.path.join(smoothed_dir, f"{i:08d}.png")
        draw_depth_map(smoothed_depth_map_path, smoothed_depth_maps[i])
    
    return list(smoothed_depth_maps)


@print_task
def save(output_dir, smoothed_depth_map_list, data_type='video', frame_names=None):
    """결과물을 영상과 파일로 저장하는 함수"""
    if data_type == 'video' and len(smoothed_depth_map_list) > 2:
        # 비디오로 저장
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_height, video_width = smoothed_depth_map_list[0].shape
        video_output = cv2.VideoWriter(os.path.join(output_dir, "depth_video.mp4"), fourcc, 30, (video_width, video_height))

        for depth_map in tqdm(smoothed_depth_map_list, desc=" - Saving depth video", total=len(smoothed_depth_map_list)):
            # 깊이 맵을 컬러 맵으로 변환
            color_depth_map = draw_depth_map(None, depth_map)
            video_output.write(color_depth_map)

        video_output.release()
        
        # npy 파일로 저장
        depth_map_numpy = np.array(smoothed_depth_map_list)
        depth_map_path = os.path.join(output_dir, "depth_map.npy")
        np.save(depth_map_path, depth_map_numpy)
        
    elif data_type == 'images':
        # 이미지 배치인 경우: 각 이미지별로 개별 디렉토리에 저장
        for idx, (depth_map, frame_name) in enumerate(tqdm(zip(smoothed_depth_map_list, frame_names), desc=" - Saving individual images", total=len(smoothed_depth_map_list))):
            frame_output_dir = os.path.join(output_dir, frame_name)
            os.makedirs(frame_output_dir, exist_ok=True)
            
            # PNG 이미지로 저장
            depth_map_png_path = os.path.join(frame_output_dir, "depth_map.png")
            draw_depth_map(depth_map_png_path, depth_map)
            
            # NPY 파일로 저장 (단일 프레임)
            depth_map_npy_path = os.path.join(frame_output_dir, "depth_map.npy")
            np.save(depth_map_npy_path, depth_map)
    else:
        # 단일 이미지인 경우
        depth_map = smoothed_depth_map_list[0]
        depth_map_path = os.path.join(output_dir, "depth_map.png")
        draw_depth_map(depth_map_path, depth_map)
        
        depth_map_numpy = np.array(smoothed_depth_map_list)
        depth_map_path = os.path.join(output_dir, "depth_map.npy")
        np.save(depth_map_path, depth_map_numpy)
        

def test(args):
    # 모델 및 프로세서 초기화
    depth_model, segment_model, segment_detail_model, device = init_model(args)
    depth_processor, segment_processor, segment_detail_processor = init_processor(args)
    
    dataset_list = args.datasets
    
    for dataset_name in dataset_list:
        # 데이터 로드
        dataset = load_data(args, dataset_name)
        
        dataset_working_dir = os.path.join(args.working_dir, dataset_name)
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_working_dir, exist_ok=True)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 영상별 처리
        for video_data in dataset:
            video_name = video_data['name']
            data_type = video_data.get('type', 'video')
            if video_name in []:
                continue
            frames = video_data['frames']
            frame_names = video_data.get('frame_names', None)
            video_working_dir = os.path.join(dataset_working_dir, video_name)
            video_output_dir = os.path.join(dataset_output_dir, video_name)
            os.makedirs(video_working_dir, exist_ok=True)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # 체크포인트 로드 또는 추론 수행
            if os.path.exists(os.path.join(video_working_dir, args.infer_result)):
                rich.print(f"[bold yellow][!] Inference results for {video_name} already exist. Skipping...[/bold yellow]")
                with open(os.path.join(video_working_dir, "infer_result.pkl"), "rb") as f:
                    infer_result = pickle.load(f)
                frame_list = infer_result['frame']
                depth_map_list = infer_result['depth_map']
                segment_map_list = infer_result['segment_map']
                segment_detail_map_list = infer_result['segment_detail_map']
                segment_detail_info_list = infer_result['segment_detail_info']
            else:
                # 추론 결과 저장을 위한 리스트 초기화
                frame_list = []
                depth_map_list = []
                segment_map_list = []
                segment_detail_map_list = []
                segment_detail_info_list = []
                
                # 프레임별 추론 수행
                for frame_path in tqdm(frames, desc=f" - Processing {video_name}"):
                    # 이미지 로드
                    image = Image.open(frame_path).convert("RGB")
                    frame_list.append(image)
                    
                    # 추론 수행
                    depth_map = infer_depth(depth_model, depth_processor, image, device) # 깊이 추정
                    segment_map = infer_segment(segment_model, segment_processor, image, device) # 세그멘테이션 (요소와 배경 구분)
                    segment_detail_map, segments_info = infer_segment_detail(segment_detail_model, segment_detail_processor, image, device) # 세그멘테이션 상세 (객체별 분할)
                    
                    depth_map_list.append(depth_map)
                    segment_map_list.append(segment_map)
                    segment_detail_map_list.append(segment_detail_map)
                    segment_detail_info_list.append(segments_info)
                
                # 추론 결과 저장    
                infer_result = {
                    'frame': frame_list,
                    'depth_map': depth_map_list,
                    'segment_map': segment_map_list,
                    'segment_detail_map': segment_detail_map_list,
                    'segment_detail_info': segment_detail_info_list
                }
                with open(os.path.join(video_working_dir, args.infer_result), "wb") as f:
                    pickle.dump(infer_result, f)

            save_stage_outputs(dataset_output_dir, video_name, 'raw', depth_map_list, data_type, frame_names)

            # 시각화
            if args.visualize:
                visualize_results(video_working_dir, frame_list, depth_map_list, segment_map_list, segment_detail_map_list, segment_detail_info_list)
                
            # method a) 원본 깊이 맵 npy 저장
            
            # 소실점 추정
            vanishing_point_list = estimate_vanishing_point(depth_map_list, num_layers=args.num_layer)
            
            # 분리 처리
            if os.path.exists(os.path.join(video_working_dir, args.separated_result)):
                rich.print(f"[bold yellow][!] Separated results for {video_name} already exist. Skipping...[/bold yellow]")
                with open(os.path.join(video_working_dir, args.separated_result), "rb") as f:
                    separated_result = pickle.load(f)
                separated_depth_map_list = separated_result['separated_depth_map']
            else:
                separated_depth_map_list = separate(video_working_dir, depth_map_list, segment_map_list, vanishing_point_list)
                separated_result = {
                    'separated_depth_map': separated_depth_map_list
                }
                with open(os.path.join(video_working_dir, args.separated_result), "wb") as f:
                    pickle.dump(separated_result, f)
                    
            # method b) 분리 처리된 깊이 맵 npy 저장
            save_stage_outputs(dataset_output_dir, video_name, 'separated', separated_depth_map_list, data_type, frame_names)
            
            # 향상 처리
            if os.path.exists(os.path.join(video_working_dir, args.enhanced_result)):
                rich.print(f"[bold yellow][!] Enhanced results for {video_name} already exist. Skipping...[/bold yellow]")
                with open(os.path.join(video_working_dir, args.enhanced_result), "rb") as f:
                    enhanced_result = pickle.load(f)
                enhanced_depth_map_list = enhanced_result['enhanced_depth_map']
            else:
                enhanced_depth_map_list = enhance(video_working_dir, separated_depth_map_list, segment_detail_map_list, segment_detail_info_list, vanishing_point_list)
                enhanced_result = {
                    'enhanced_depth_map': enhanced_depth_map_list
                }
                with open(os.path.join(video_working_dir, args.enhanced_result), "wb") as f:
                    pickle.dump(enhanced_result, f)
                    
            # method c) 향상 처리된 깊이 맵 npy 저장
            save_stage_outputs(dataset_output_dir, video_name, 'enhanced', enhanced_depth_map_list, data_type, frame_names)
            
            # 시간적 일관성 처리 (비디오인 경우만)
            if data_type == 'video':
                if os.path.exists(os.path.join(video_working_dir, args.smoothed_result)):
                    rich.print(f"[bold yellow][!] Smoothed results for {video_name} already exist. Skipping...[/bold yellow]")
                    with open(os.path.join(video_working_dir, args.smoothed_result), "rb") as f:
                        smoothed_result = pickle.load(f)
                    smoothed_depth_map_list = smoothed_result['smoothed_depth_map']
                else:
                    smoothed_depth_map_list = temporal_smooth(video_working_dir, enhanced_depth_map_list, lambda_val=0.1, num_iterations=2)
                    smoothed_result = {
                        'smoothed_depth_map': smoothed_depth_map_list
                    }
                    with open(os.path.join(video_working_dir, args.smoothed_result), "wb") as f:
                        pickle.dump(smoothed_result, f)
            else:
                # 이미지 배치인 경우 시간적 스무딩 건너뛰기
                rich.print(f"[bold yellow][!] Skipping temporal smoothing for images batch {video_name}[/bold yellow]")
                smoothed_depth_map_list = enhanced_depth_map_list
                
            # method d) 시간적 일관성 처리된 깊이 맵 npy 저장
            save_stage_outputs(dataset_output_dir, video_name, 'smoothed', smoothed_depth_map_list, data_type, frame_names)
            
            # 결과 저장
            save(video_output_dir, smoothed_depth_map_list, data_type=data_type, frame_names=frame_names)
            rich.print(f"[bold green][*] Results for {video_name} saved to {video_output_dir}[/bold green]\n")
            
def preprocess_depth_map(dataset, dataset_name):
    """깊이 맵 전처리 함수"""
    for idx, data in tqdm(enumerate(dataset), desc=" - Preprocessing", total=len(dataset)):
        pred = data['pred'].astype(np.float64, copy=True)
        gt = data['gt'].astype(np.float64, copy=True)

        # Dataset-specific GT normalization
        if dataset_name == 'nyu':
            gt = gt / 1000.0  # millimeter -> meter
            gt_min_clip, gt_max_clip = 0.5, 10.0
        elif dataset_name == 'kitti':
            gt = gt / 256.0  # KITTI depth PNG scale -> meter
            gt_min_clip, gt_max_clip = 1.0, 80.0
        else:
            gt_min_clip, gt_max_clip = 0.0, np.inf

        # Clamp GT to evaluation range and mask invalid pixels
        gt_valid_mask = np.isfinite(gt) & (gt > 0)
        range_mask = (gt >= gt_min_clip) & (gt <= gt_max_clip)
        gt_valid_mask &= range_mask
        gt = np.clip(gt, gt_min_clip, gt_max_clip)
        gt[~gt_valid_mask] = np.nan

        # Prepare valid pixel pairs
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        valid_mask = np.isfinite(pred_flat) & np.isfinite(gt_flat) & (gt_flat > 0)

        if np.count_nonzero(valid_mask) < 50:
            dataset[idx]['pred'] = np.full_like(pred, np.nan)
            dataset[idx]['gt'] = gt
            continue

        pred_valid = pred_flat[valid_mask]
        gt_valid = gt_flat[valid_mask]

        # Solve for optimal scale & shift (least squares) as in SOTA monocular depth eval
        A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
        try:
            scale, shift = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
        except np.linalg.LinAlgError:
            scale, shift = 1.0, 0.0

        pred_aligned = scale * pred + shift
        pred_aligned = np.clip(pred_aligned, gt_min_clip, gt_max_clip)

        # Keep invalid regions as NaN so they are ignored later
        pred_result = np.full_like(pred_aligned, np.nan)
        pred_result_flat = pred_result.flatten()
        pred_aligned_flat = pred_aligned.flatten()
        pred_result_flat[valid_mask] = pred_aligned_flat[valid_mask]
        pred_result = pred_result_flat.reshape(pred.shape)

        dataset[idx]['pred'] = pred_result
        dataset[idx]['gt'] = gt
            
    return dataset


def _extract_valid_pairs(pred, gt, require_positive_pred=False):
    """평가에 사용할 유효한 (pred, gt) 쌍 추출"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid_mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if require_positive_pred:
        valid_mask &= (pred > 0)
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    return pred_valid, gt_valid

def calculate_absral(pred, gt):
    """AbsRel 계산 함수 (NaN 값 제외)"""
    pred_valid, gt_valid = _extract_valid_pairs(pred, gt)
    if len(pred_valid) == 0:
        return float('nan')
    return np.mean(np.abs(gt_valid - pred_valid) / gt_valid)

def calculate_sqrel(pred, gt):
    """SqRel 계산 함수 (NaN 값 제외)"""
    pred_valid, gt_valid = _extract_valid_pairs(pred, gt)
    if len(pred_valid) == 0:
        return float('nan')
    return np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)

def calculate_rmselog(pred, gt):
    """RMSLog 계산 함수 (NaN 값 제외)"""
    pred_valid, gt_valid = _extract_valid_pairs(pred, gt, require_positive_pred=True)
    if len(pred_valid) == 0:
        return float('nan')
    eps = 1e-6
    log_diff = np.log(np.clip(pred_valid, eps, None)) - np.log(np.clip(gt_valid, eps, None))
    return np.sqrt(np.mean(log_diff ** 2))

def calculate_delta(pred, gt, threshold=1.25):
    """Delta accuracy 계산 함수 (NaN 값 제외)"""
    pred_valid, gt_valid = _extract_valid_pairs(pred, gt, require_positive_pred=True)
    if len(pred_valid) == 0:
        return float('nan')
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    return np.mean(ratio < threshold)
            
def calculate(args):
    dataset_list = args.datasets
    stage_list = ['raw', 'separated', 'enhanced', 'smoothed']
    
    for dataset_name in dataset_list:
        for stage_name in stage_list:
            dataset = load_calc_data(args, dataset_name, stage_name=stage_name)
            if len(dataset) == 0:
                continue
            dataset = preprocess_depth_map(dataset, dataset_name)
            result_path = os.path.join(args.output_dir, f"{dataset_name}_{stage_name}_calculation_results.csv")
            
            header = ["Index", "AbsRel", "SqRel", "RMSLog", "Delta1", "Delta2", "Delta3"]
            with open(result_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
            
            absrel_list = []
            sqrel_list = []
            rmselog_list = []
            delta1_list = []
            delta2_list = []
            delta3_list = []
            for idx, data in tqdm(enumerate(dataset), desc=f" - Calculating metrics for {dataset_name} ({stage_name})", total=len(dataset)):
                pred = data['pred']
                gt = data['gt']
                
                absrel = calculate_absral(pred, gt)
                sqrel = calculate_sqrel(pred, gt)
                rmselog = calculate_rmselog(pred, gt)
                delta1 = calculate_delta(pred, gt, threshold=1.25)
                delta2 = calculate_delta(pred, gt, threshold=1.25 ** 2)
                delta3 = calculate_delta(pred, gt, threshold=1.25 ** 3)
                with open(result_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([idx, absrel, sqrel, rmselog, delta1, delta2, delta3])
                
                absrel_list.append(absrel)
                sqrel_list.append(sqrel)
                rmselog_list.append(rmselog)
                delta1_list.append(delta1)
                delta2_list.append(delta2)
                delta3_list.append(delta3)

            def safe_mean(values):
                values = np.array(values, dtype=np.float64)
                return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float('nan')

            mean_absrel = safe_mean(absrel_list)
            mean_sqrel = safe_mean(sqrel_list)
            mean_rmselog = safe_mean(rmselog_list)
            mean_delta1 = safe_mean(delta1_list)
            mean_delta2 = safe_mean(delta2_list)
            mean_delta3 = safe_mean(delta3_list)

            with open(result_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["mean", mean_absrel, mean_sqrel, mean_rmselog, mean_delta1, mean_delta2, mean_delta3])

            rich.print(
                f"[bold cyan]Dataset {dataset_name} ({stage_name}) mean metrics[/bold cyan] "
                f"AbsRel: {mean_absrel:.4f}, SqRel: {mean_sqrel:.4f}, RMSLog: {mean_rmselog:.4f}, "
                f"Delta1: {mean_delta1:.4f}, Delta2: {mean_delta2:.4f}, Delta3: {mean_delta3:.4f}"
            )

if __name__ == "__main__":
    args = init_args()
    test(args)
    calculate(args)