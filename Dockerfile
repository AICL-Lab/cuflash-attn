# CuFlash-Attn Docker Image
# Provides a ready-to-use environment for building and testing

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL maintainer="CuFlash-Attn Team"
LABEL description="CuFlash-Attn: High-performance CUDA FlashAttention implementation"

ARG CMAKE_VERSION=3.28.1
ARG GCC_VERSION=11

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    wget \
    git \
    curl \
    ca-certificates \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install specific GCC version
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-${GCC_VERSION} \
    g++-${GCC_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 100

# Install CMake
RUN cd /tmp && \
    wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh" && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# Install Python packages for testing/benchmarking
RUN pip3 install --no-cache-dir \
    numpy \
    torch \
    pytest \
    matplotlib \
    pandas

# Set working directory
WORKDIR /workspace/cuflash-attn

# Copy the entire project
COPY . .

# Build the project
RUN cmake --preset release && \
    cmake --build --preset release --parallel $(nproc)

# Set environment variables
ENV PYTHONPATH=/workspace/cuflash-attn:$PYTHONPATH
ENV PATH=/workspace/cuflash-attn/build/release:$PATH

# Default command
CMD ["/bin/bash"]
