# STF-Depth: Semantic and Temporal Fusion Depth Estimation

**STF-Depth**는 Semantic and Temporal Fusion Depth Estimation의 약자로, 단일 이미지에서 발생하는 깊이 추정의 부정확성을 개선하는 것을 목표로 하는 파이프라인입니다.

이 프로젝트는 동영상 도메인이 가지는 \*\*시간적 연속성(Temporal Fusion)\*\*과 \*\*세그멘테이션을 통한 의미/형태 정보(Semantic Fusion)\*\*를 활용하여, 프레임 간 일관성을 높이고 더 사실적인 깊이 정보를 생성합니다.

-----

## 주요 기능

  * **다중 모델 파이프라인:** 최신 딥러닝 모델을 활용하여 비디오의 각 프레임에 대한 깊이 및 세그멘테이션 맵을 생성합니다.
      * **깊이 추정 (Depth Estimation):** MiDaS (DPT-Large)
      * **시맨틱 분할 (Semantic Segmentation):** DeepLabV3
      * **패놉틱 분할 (Panoptic Segmentation):** OneFormer
  * **자동화된 처리:** 입력 비디오 폴더를 지정하면 프레임 추출부터 모델 추론, 결과 저장까지 전 과정을 자동으로 수행합니다.
  * **결과 캐싱:** 이미 처리된 비디오는 중간 결과를 저장(`.pkl`)하여, 재실행 시 추론 과정을 건너뛰고 빠르게 시각화하거나 추가 작업을 수행할 수 있습니다.
  * **시각화:** 각 모델의 출력 결과를 이미지 파일로 저장하여 직관적으로 확인할 수 있습니다.

-----

## 설치 (Installation)

이 프로젝트를 실행하기 위한 모든 의존성은 Conda 가상 환경을 통해 관리됩니다.

### 1\. Conda 환경 생성 및 활성화

프로젝트를 위한 `sftdepth`라는 이름의 Conda 가상 환경을 생성하고 활성화합니다.

```bash
# 1. 'sftdepth'라는 이름으로 Python 3.10 환경 생성
conda create -n sftdepth python=3.10

# 2. 생성한 환경 활성화
conda activate sftdepth
```

### 2\. 필수 라이브러리 설치

가장 중요한 PyTorch를 CUDA와 함께 설치한 후, 나머지 라이브러리들을 설치합니다.

```bash
# 1. PyTorch 설치 (CUDA 11.8 기준)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 2. 주요 라이브러리 설치
pip install transformers opencv-python rich matplotlib tqdm timm
```

### (대안) `environment.yml` 파일로 한 번에 설치

프로젝트 루트에 아래 내용으로 `environment.yml` 파일을 생성한 뒤, 다음 명령어를 실행하여 한 번에 환경을 구축할 수도 있습니다.

```yaml
# environment.yml
name: sftdepth
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - pip
  - pip:
    - transformers
    - opencv-python
    - rich
    - matplotlib
    - tqdm
    - timm
```

```bash
# yml 파일을 이용해 환경 생성
conda env create -f environment.yml
```

-----

## 사용법 (Usage)

1.  **입력 데이터 준비:** `data/input` 폴더 안에 분석하고 싶은 비디오 파일(`.mp4`, `.avi` 등)을 넣습니다.

2.  **스크립트 실행:** 터미널에서 아래 명령어를 실행합니다.

    ```bash
    # Conda 환경 활성화
    conda activate sftdepth

    # 스크립트 실행
    python run.py
    ```

### 명령줄 인자 (Arguments)

  * `--input_dir`: 입력 비디오가 있는 디렉터리 (기본값: `./data/input`)
  * `--output_dir`: 최종 결과물이 저장될 디렉터리 (기본값: `./data/output`)
  * `--working_dir`: 프레임, 중간 결과(.pkl), 시각화 이미지 등이 저장될 작업 디렉터리 (기본값: `./data/working`)
  * `--visualize`: 이 플래그를 사용하면 시각화 결과를 저장하지 않습니다. (기본값: 저장함)

-----

## 📂 프로젝트 구조

```
.
├── data
│   ├── input/                # (입력) 비디오 파일을 이곳에 넣으세요.
│   ├── working/              # (중간 결과) 프레임, .pkl, 시각화 결과 저장
│   │   └── [video_name]/
│   │       ├── frames/
│   │       ├── depth/
│   │       ├── segment/
│   │       ├── segment_detail/
│   │       └── infer_result.pkl
│   └── output/               # (최종 결과) 최종 결과물 저장 (현재 코드에서는 사용되지 않음)
├── run.py                    # 메인 실행 스크립트
└── README.md                 # 프로젝트 설명 파일
```