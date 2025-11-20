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
from sklearn.linear_model import RANSACRegressor

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
def estimate_vanishing_point(depth_map_list, num_layers=5):
    """소실점 추정 함수"""
    def estimate(depth_map, num_layers):
        """깊이맵에서 소실점 추정 함수"""
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
            
    vanishing_point_list = []
    
    for depth_map in depth_map_list:
        vp = estimate(depth_map, num_layers)
        vanishing_point_list.append(vp)
    
    return vanishing_point_list


@print_task
def separate(working_dir, depth_map_list, segment_map_list, vanishing_point_list):
    """배경과 전경의 구분을 강화하는 함수"""
    separated_dir = os.path.join(working_dir, "separated_depth")
    os.makedirs(separated_dir, exist_ok=True)
    separated_depth_map_list = []
    for idx, (depth_map, segment_map, vp) in tqdm(enumerate(zip(depth_map_list, segment_map_list, vanishing_point_list)), desc=f" - Separating depth map", total=len(depth_map_list)):
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
            slope, intercept = estimate_offset_equation(separated_depth_map, vp)
            for y in range(h):
                for x in range(w):
                    if binary_mask[y, x] == 1:
                        offset = slope * separated_depth_map[y, x] + intercept
                        separated_depth_map[y, x] += offset # 배경 깊이 조정
        
        separated_depth_map_path = os.path.join(separated_dir, f"{idx:08d}.png")
        draw_depth_map(separated_depth_map_path, separated_depth_map)
        
        separated_depth_map_list.append(separated_depth_map)
    return separated_depth_map_list

@print_task
def enhance(working_dir, separated_depth_map_list, segment_detail_map_list, segment_detail_info_list, vanishing_point_list):
    """객체별 깊이 구분을 강화하는 함수"""
    enhanced_dir = os.path.join(working_dir, "enhanced_depth")
    os.makedirs(enhanced_dir, exist_ok=True)
    enhanced_depth_map_list = []
    
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
    
    for idx, (separated_depth_map, segment_detail_map, segment_detail_info, vp) in tqdm(enumerate(zip(separated_depth_map_list, segment_detail_map_list, segment_detail_info_list, vanishing_point_list)), desc=f" - Enhancing depth map", total=len(separated_depth_map_list)):
        enhanced_depth_map = separated_depth_map.copy()
        h, w = enhanced_depth_map.shape
        panoptic_map_np = segment_detail_map.cpu().numpy()
        
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
        
        # 인스턴스별 데이터 수집
        instance_data = {}
        for y in range(h):
            for x in range(w):
                segment_id = panoptic_map_np[y, x]
                depth = enhanced_depth_map[y, x]
                if segment_id not in instance_data:
                    instance_data[segment_id] = {
                        'class_id': info[segment_id]['label_id'] if segment_id in info else -1,
                        'class_name': label_map.get(str(info[segment_id]['label_id']), "unknown") if segment_id in info else "unknown",
                        'score': info[segment_id]['score'] if segment_id in info else 0.0,
                        'depths': [],
                        'positions': [],
                        'min_x': w,
                        'max_x': 0,
                        'min_y': h,
                        'max_y': 0
                    }
                instance_data[segment_id]['depths'].append(depth)
                instance_data[segment_id]['positions'].append((y, x))
                instance_data[segment_id]['min_x'] = min(instance_data[segment_id]['min_x'], x)
                instance_data[segment_id]['max_x'] = max(instance_data[segment_id]['max_x'], x)
                instance_data[segment_id]['min_y'] = min(instance_data[segment_id]['min_y'], y)
                instance_data[segment_id]['max_y'] = max(instance_data[segment_id]['max_y'], y)
        
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
        
        # z-index 할당
        z_index_map = {item['id']: z for z, item in enumerate(instance_list)}
        
        # 겹침 감지 및 분리 강화
        for i, instance_i in enumerate(instance_list):
            for j in range(i + 1, len(instance_list)):
                instance_j = instance_list[j]
                
                # y 좌표가 겹치는지 확인
                if not (instance_i['max_y'] < instance_j['min_y'] or instance_j['max_y'] < instance_i['min_y']):
                    # 겹침이 발견됨 - instance_j가 더 가까우므로 깊이 차이 강화
                    depth_diff = instance_j['mean_depth'] - instance_i['mean_depth']
                    if depth_diff > 0:
                        # 겹친 영역의 깊이 차이를 더 명확하게
                        overlap_factor = 1.5
                        for pos in instance_j['positions']:
                            y, x = pos
                            enhanced_depth_map[y, x] += depth_diff * overlap_factor
        
        # 인스턴스별 offset 적용
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
            
            # 인스턴스의 모든 픽셀에 offset 적용
            for pos in instance['positions']:
                y, x = pos
                enhanced_depth_map[y, x] += final_offset
                    
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