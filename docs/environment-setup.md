## CTSF-W 실험 환경 세팅 가이드 (RunPod, CUDA 11.8)

이 문서는 순수 Ubuntu + CUDA 11.8 기반의 RunPod 인스턴스에서 CTSF-W 레포지토리와 동일한 실험 환경을 재현하기 위한 지침입니다.  
현재 개발 환경 기준 주요 스펙은 아래와 같습니다.

- OS: Ubuntu (RunPod 기본 이미지)
- Python: 3.11.11
- CUDA Toolkit: 11.8 (드라이버/런타임)
- PyTorch: 2.7.1+cu118
- NumPy: 2.1.2

---

### 1. CUDA 11.8 준비

RunPod CUDA 11.8 이미지에는 드라이버가 기본 포함되어 있습니다.  
자체 설치가 필요한 경우 공식 안내를 참고하십시오: <https://developer.nvidia.com/cuda-11-8-0-download-archive>

필수 확인 명령:
```bash
nvidia-smi
nvcc --version
```

---

### 2. Conda 환경 생성

Miniforge 또는 Miniconda를 설치한 뒤, CTSF 전용 환경을 생성합니다.
```bash
conda create -n CTSF python=3.11
conda activate CTSF
```

---

### 3. PyTorch + CUDA 11.8 설치

PyTorch 2.7.1 및 관련 패키지를 CUDA 11.8 전용 휠에서 설치합니다.
```bash
pip install --upgrade pip
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

> 위 명령은 `nvidia-cublas-cu11`, `nvidia-cudnn-cu11` 등 필요한 CUDA 런타임 라이브러리를 함께 설치합니다.

---

### 4. 핵심 Python 패키지 설치

실험 코드에서 직접 사용되는 주요 라이브러리 버전은 다음과 같습니다.

| 카테고리 | 패키지 | 버전 |
| --- | --- | --- |
| 수치 계산 | `numpy` | 2.1.2 |
| 과학 연산 | `scipy` | 1.16.3 |
| 머신러닝 | `scikit-learn` | 1.7.2 |
| 데이터 처리 | `pandas` | 2.3.2 |
| 시각화 | `matplotlib` | 3.10.5 |
| 유틸 | `tqdm` | 4.67.1 |
| 설정 관리 | `PyYAML` | 6.0.2 |

설치 예시:
```bash
pip install numpy==2.1.2 scipy==1.16.3 scikit-learn==1.7.2 \
    pandas==2.3.2 matplotlib==3.10.5 tqdm==4.67.1 PyYAML==6.0.2
```

추가로, 코드에서 참조되는 유틸 패키지들도 설치합니다.
```bash
pip install fsspec==2024.6.1 filelock==3.13.1 requests==2.32.5 \
    psutil==7.0.0 typing_extensions==4.15.0
```

---

### 5. 노트북/실험 편의 패키지 (선택)

Jupyter 기반 실험이나 Notebook 활용 시:
```bash
pip install jupyterlab==4.4.6 notebook==7.4.5 ipywidgets==8.1.7 \
    ipykernel==6.30.1
```

---

### 6. 패키지 전체 목록 (참고)

현재 개발 환경에서 `pip list --format=freeze`로 추출한 전체 목록은 `env_package_list.txt`에 저장되어 있습니다.  
필요 시 다음 명령으로 동일한 버전을 한 번에 설치할 수 있습니다.
```bash
pip install -r env_package_list.txt
```

다만, 상기 파일에는 노트북/유틸용 패키지가 모두 포함되어 있으므로, 최소 구성만 필요하다면 위의 핵심 패키지 설치 절차만 따르면 됩니다.

---

### 7. 설치 확인

환경 세팅이 완료되면 아래 명령으로 핵심 라이브러리 버전을 확인합니다.
```bash
python -c "import torch, numpy, pandas; \
print('torch', torch.__version__); \
print('numpy', __import__('numpy').__version__); \
print('pandas', __import__('pandas').__version__)"
```

이상이면 CTSF-W 레포지토리에서 제공하는 실험 (`run_suite.py`, `run_all_experiments.py`, `test_single_experiments.py`)을 동일한 환경에서 실행할 준비가 끝났습니다.


