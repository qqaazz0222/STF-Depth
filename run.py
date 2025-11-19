import os
import time
import rich
import pickle
from tqdm import tqdm
from functools import wraps
from argparse import ArgumentParser
from collections import defaultdict
import warnings
import logging

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from transformers import AutoProcessor, AutoModelForUniversalSegmentation

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
    parser.add_argument('--input_dir', type=str, default="./data/input", help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default="./data/output", help='Output directory to save results')
    parser.add_argument('--working_dir', type=str, default="./data/working", help='Working directory for intermediate files')
    parser.add_argument('--depth_model', type=list, default=["intel-isl/MiDaS", "DPT_Large"], help='Depth estimation model name')
    parser.add_argument('--depth_utils', type=list, default=["intel-isl/MiDaS", "transforms"], help='Depth estimation utilities name')
    parser.add_argument('--segment_model', type=list, default=['pytorch/vision:v0.10.0', 'deeplabv3_resnet101'], help='Segmentation model name')
    parser.add_argument('--segment_detail_model', type=list, default=['shi-labs/oneformer_coco_swin_large'], help='Segmentation model name')
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
def load_data(args):
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
            cv2.imwrite(frame_path, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        vidcap.release()
        return frame_list
    
    input_dir = args.input_dir
    working_dir = args.working_dir
    ext_list = ['.mp4', '.avi', '.mov']
    
    dataset = []
    video_list = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.lower().endswith(tuple(ext_list))]
    video_list.sort()
    for video_path in tqdm(video_list, desc=" - Loading videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cur_video_working_dir = os.path.join(working_dir, video_name)
        cur_video_frame_dir = os.path.join(cur_video_working_dir, "frames")
        os.makedirs(cur_video_working_dir, exist_ok=True)
        os.makedirs(cur_video_frame_dir, exist_ok=True)
        frame_list = extract_video_to_frames(video_path, cur_video_frame_dir)
        dataset.append({'name': video_name, 'path': video_path, 'frames': frame_list})
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
    colored_map = (plt.cm.viridis_r(normalized_map)[:, :, :3] * 255).astype(np.uint8)
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

    for idx, (image, segment_map) in tqdm(enumerate(zip(frame_list, segment_map_list)), desc=f" - Visualizing segment map", total=len(frame_list)):
        segment_map_path = os.path.join(segment_map_dir, f"{idx:08d}.png")
        draw_segment_map(segment_map_path, image, segment_map)
        
    for idx, (image, segment_detail_map, segments_info) in tqdm(enumerate(zip(frame_list, segment_detail_map_list, segment_detail_info_list)), desc=f" - Visualizing segment detail map", total=len(frame_list)):
        segment_detail_map_path = os.path.join(segment_detail_map_dir, f"{idx:08d}.png")
        draw_panoptic_map(segment_detail_map_path, image, segment_detail_map, segments_info)
    
    return None

@print_task
def separate(working_dir, depth_map_list, segment_map_list):
    """배경과 전경의 구분을 강화하는 함수"""
    separated_dir = os.path.join(working_dir, "separated_depth")
    os.makedirs(separated_dir, exist_ok=True)
    separated_depth_map_list = []
    for idx, (depth_map, segment_map) in tqdm(enumerate(zip(depth_map_list, segment_map_list)), desc=f" - Separating depth map", total=len(depth_map_list)):
        separated_depth_map = depth_map.copy()
        h, w = separated_depth_map.shape
        binary_mask = (segment_map != 0).cpu().numpy().astype(np.float32)  # 배경(0)과 요소(1) 구분
        mask_uint8 = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bottom_point_list = []
        
        for contour in contours:
            if len(contour) > 0:
                bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
                bottom_point_list.append(bottom_point)
                
        if bottom_point_list: 
            mean_y = np.mean([pt[1] for pt in bottom_point_list]) if bottom_point_list else None
            for y in range(h):
                gap = mean_y - y
                for x in range(w):
                    if binary_mask[y, x] == 0:
                        separated_depth_map[y, x] -= gap * 0.025 # 배경 깊이 조정
            for contour in contours:
                bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
                x = int(bottom_point[0])
                y = int(bottom_point[1])
                floor_y = y + 1
                if floor_y >= h:
                    offset = 1
                else:
                    bottom_depth = separated_depth_map[y, x]
                    floor_depth = separated_depth_map[floor_y, x]
                    offset = floor_depth - bottom_depth + 1 # 바닥과 요소 사이의 깊이 차이 + 여유
                
                mask = np.zeros_like(binary_mask, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
                separated_depth_map[mask == 1] += offset
        
        separated_depth_map_path = os.path.join(separated_dir, f"{idx:08d}.png")
        draw_depth_map(separated_depth_map_path, separated_depth_map)
        
        separated_depth_map_list.append(separated_depth_map)
    return separated_depth_map_list

@print_task
def enhance(working_dir, separated_depth_map_list, segment_detail_map_list):
    """객체별 깊이 구분을 강화하는 함수"""
    enhanced_dir = os.path.join(working_dir, "enhanced_depth")
    os.makedirs(enhanced_dir, exist_ok=True)
    enhanced_depth_map_list = []
    
    for idx, (separated_depth_map, segment_detail_map) in tqdm(enumerate(zip(separated_depth_map_list, segment_detail_map_list)), desc=f" - Enhancing depth map", total=len(separated_depth_map_list)):
        enhanced_depth_map = separated_depth_map.copy()
        h, w = enhanced_depth_map.shape
        panoptic_map_np = segment_detail_map.cpu().numpy()
        
        z_index_data = {}
        
        for y in range(h):
            for x in range(w):
                segment_id = panoptic_map_np[y, x]
                depth = enhanced_depth_map[y, x]
                if segment_id not in z_index_data:
                    z_index_data[segment_id] = []
                z_index_data[segment_id].append(depth)

        z_index_list = [[int(z_index_item[0]), float(np.mean(z_index_item[1]))] for z_index_item in z_index_data.items()]
        z_index_list.sort(key=lambda item: item[1], reverse=True)
        
        z_index_map = {item[0]: z for z, item in enumerate(z_index_list)}
        
        for y in range(h):
            for x in range(w):
                segment_id = panoptic_map_np[y, x]
                z_index = z_index_map.get(segment_id, None)
                if z_index is not None:
                    enhanced_depth_map[y, x] += z_index * 0.3  # 객체별 깊이 조정 (z-index에 비례)
                    
        enhanced_depth_map_path = os.path.join(enhanced_dir, f"{idx:08d}.png")
        draw_depth_map(enhanced_depth_map_path, enhanced_depth_map)
        
        enhanced_depth_map_list.append(enhanced_depth_map)
    return enhanced_depth_map_list


@print_task
def temporal_smooth(video_working_dir, enhanced_depth_map_list, lambda_val=0.5, num_iterations=2):
    """시간적 일관성을 위한 깊이 맵 스무딩 함수"""
    smoothed_dir = os.path.join(video_working_dir, "smoothed_depth")
    os.makedirs(smoothed_dir, exist_ok=True)
    num_frames = len(enhanced_depth_map_list)
    smoothed_depth_map_list = [m.copy() for m in enhanced_depth_map_list]

    for _ in range(num_iterations):
        new_smoothed_list = []
        for i in tqdm(range(num_frames), desc=f" - Smoothing depth map, iteration {_ + 1}/{num_iterations}", total=num_frames):
            if i == 0:
                new_map = (1 - lambda_val) * smoothed_depth_map_list[i] + lambda_val * smoothed_depth_map_list[i + 1]
            elif i == num_frames - 1:
                new_map = (1 - lambda_val) * smoothed_depth_map_list[i] + lambda_val * smoothed_depth_map_list[i - 1]
            else:
                new_map = (1 - 2 * lambda_val) * smoothed_depth_map_list[i] + lambda_val * (smoothed_depth_map_list[i - 1] + smoothed_depth_map_list[i + 1])
            new_smoothed_list.append(new_map)
        smoothed_depth_map_list = new_smoothed_list

    for i, new_map in enumerate(smoothed_depth_map_list):
        smoothed_depth_map_path = os.path.join(smoothed_dir, f"{i:08d}.png")
        draw_depth_map(smoothed_depth_map_path, new_map)
        
    return smoothed_depth_map_list


@print_task
def save(output_dir, smoothed_depth_map_list):
    """결과물을 영상과 파일로 저장하는 함수"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_height, video_width = smoothed_depth_map_list[0].shape
    video_output = cv2.VideoWriter(os.path.join(output_dir, "depth_video.mp4"), fourcc, 30, (video_width, video_height))

    for depth_map in tqdm(smoothed_depth_map_list, desc=" - Saving depth video", total=len(smoothed_depth_map_list)):
        # 깊이 맵을 컬러 맵으로 변환
        color_depth_map = draw_depth_map(None, depth_map)
        video_output.write(color_depth_map)

    video_output.release()
    
    depth_map_numpy = np.array(smoothed_depth_map_list)
    depth_map_path = os.path.join(output_dir, "depth_map.npy")
    np.save(depth_map_path, depth_map_numpy)


def main(args):
    # 모델 및 프로세서 초기화
    depth_model, segment_model, segment_detail_model, device = init_model(args)
    depth_processor, segment_processor, segment_detail_processor = init_processor(args)
    
    # 데이터 로드
    dataset = load_data(args)
    
    # 영상별 처리
    for video_data in dataset:
        video_name = video_data['name']
        if video_name in []:
            continue
        frames = video_data['frames']
        video_working_dir = os.path.join(args.working_dir, video_name)
        video_output_dir = os.path.join(args.output_dir, video_name)
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

        # 시각화
        if args.visualize:
            visualize_results(video_working_dir, frame_list, depth_map_list, segment_map_list, segment_detail_map_list, segment_detail_info_list)
        
        # 분리 처리
        if os.path.exists(os.path.join(video_working_dir, args.separated_result)):
            rich.print(f"[bold yellow][!] Separated results for {video_name} already exist. Skipping...[/bold yellow]")
            with open(os.path.join(video_working_dir, args.separated_result), "rb") as f:
                separated_result = pickle.load(f)
            separated_depth_map_list = separated_result['separated_depth_map']
        else:
            separated_depth_map_list = separate(video_working_dir, depth_map_list, segment_map_list)
            separated_result = {
                'separated_depth_map': separated_depth_map_list
            }
            with open(os.path.join(video_working_dir, args.separated_result), "wb") as f:
                pickle.dump(separated_result, f)
        
        # 향상 처리
        if os.path.exists(os.path.join(video_working_dir, args.enhanced_result)):
            rich.print(f"[bold yellow][!] Enhanced results for {video_name} already exist. Skipping...[/bold yellow]")
            with open(os.path.join(video_working_dir, args.enhanced_result), "rb") as f:
                enhanced_result = pickle.load(f)
            enhanced_depth_map_list = enhanced_result['enhanced_depth_map']
        else:
            enhanced_depth_map_list = enhance(video_working_dir, separated_depth_map_list, segment_detail_map_list)
            enhanced_result = {
                'enhanced_depth_map': enhanced_depth_map_list
            }
            with open(os.path.join(video_working_dir, args.enhanced_result), "wb") as f:
                pickle.dump(enhanced_result, f)
        
        # 시간적 일관성 처리
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
        
        # 결과 저장
        save(video_output_dir, smoothed_depth_map_list)
        rich.print(f"[bold green][*] Results for {video_name} saved to {video_output_dir}[/bold green]\n")

if __name__ == "__main__":
    args = init_args()
    main(args)