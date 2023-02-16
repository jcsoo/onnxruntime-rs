FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# dependencies
RUN apt update &&\
    apt install -y \
    clang \
    curl \
    libnvinfer8 \
    libnvinfer-dev \
    libnvinfer-plugin8 \
    libnvinfer-plugin-dev

# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.67.1 -y

# install onnxruntime
RUN cd /tmp &&\
    curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-gpu-1.14.0.tgz -o onnxruntime-linux-x64-gpu-1.14.0.tgz &&\
    tar xvf onnxruntime-linux-x64-gpu-1.14.0.tgz -C /opt &&\
    ln -s /opt/onnxruntime-linux-x64-gpu-1.14.0 /opt/onnxruntime &&\
    rm onnxruntime-linux-x64-gpu-1.14.0.tgz
ENV ORT_LIB_LOCATION=/opt/onnxruntime
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/onnxruntime/lib
ENV CUDA_MODULE_LOADING=LAZY