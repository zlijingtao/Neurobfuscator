# FROM octoml/tvm:latest-gpu
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

# Install deps
ENV DEBIAN_FRONTEND=nonintercative
RUN apt update && apt install -y --no-install-recommends software-properties-common && apt update
RUN apt install -y --no-install-recommends gcc make build-essential wget tar cmake


ENV DEBIAN_FRONTEND=nonintercative
RUN add-apt-repository ppa:deadsnakes/ppa && apt update 

RUN apt install -y --no-install-recommends zlib1g-dev libedit-dev libxml2-dev nano git libgtest-dev unzip libtinfo-dev libz-dev \
    libcurl4-openssl-dev libopenblas-dev g++ sudo python3.6 python3-setuptools python3-pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

RUN update-alternatives  --set python /usr/bin/python3.6

RUN python -V

RUN apt update

RUN python -m pip install pylint==1.9.4 six numpy pytest cython decorator scipy tornado gym pandas GPUtil matplotlib

RUN python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN python -m pip install tensorflow==1.13.1 xgboost sklearn psutil

# LLVM
RUN echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal-10 main \
     >> /etc/apt/sources.list.d/llvm.list && \
     wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
     apt-get update && apt-get install -y llvm-10

RUN cd /usr && \
     git clone https://github.com/apache/incubator-tvm.git tvm --recursive && \
     cd /usr/tvm && \
     git checkout 58c3413a3 && \
     mkdir build && \
     cp cmake/config.cmake build

COPY misc/copy2tvm/tvm /usr/tvm
RUN  cd /usr/tvm/build && bash -c \
     "echo set\(USE_LLVM llvm-config-10\) >> config.cmake && \
     echo set\(USE_CUDA ON\) >> config.cmake" && \
     cmake .. && \
     make -j4 

ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/topi/python:${PYTHONPATH}

# Environment variables for CUDA
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

ENV TVM_HOME=/usr/tvm

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
RUN  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2020.3.0 && \
     rm -rf /var/lib/apt/lists/*


ENV PATH="/opt/nvidia/nsight-compute/2020.3.0:$PATH"

COPY . /root/neurob

ARG target="/root/Documents/NVIDIA Nsight Compute/2020.3.0/Sections"

COPY misc/ncu_section ${target}

WORKDIR /root/neurob/scripts
CMD ["bash"]