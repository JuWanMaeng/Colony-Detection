# -------------------------------------------------------
# BASE IMAGE: CUDA 12.6 + cuDNN + Ubuntu 22.04
# -------------------------------------------------------
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget vim && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspacess

# requirements 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt

# 코드 복사 이렇게 하면 Docker build 시점의 코드 스냅샷이 이미지에 포함돼서
# 배포용 환경에서도 코드가 같이 들어감.
# 그러나 개발 중에는 사용하지 안흔ㄴ것이 좋음 -> 매번 코드 바뀔 때마다 다시 build해야 하기 때문이다

# COPY . .

# 환경변수 (출력버퍼 flush)
ENV PYTHONUNBUFFERED=1

# 기본 실행 명령
CMD ["/bin/bash"]

