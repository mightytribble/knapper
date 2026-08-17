# syntax=docker/dockerfile:1
#
# VARIANT selects the base image and cargo features:
#   cpu  -> ubuntu base, default features (~100 MB image)
#   cuda -> nvidia/cuda devel base, --features cuda, fat GPU binary (~700 MB)
ARG VARIANT=cpu
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG CPU_IMAGE=ubuntu:24.04

FROM ${CPU_IMAGE} AS base-cpu
ENV KNAPPER_CARGO_FEATURES=""
ENV CUDAARCHS=""

FROM ${CUDA_IMAGE} AS base-cuda
ENV KNAPPER_CARGO_FEATURES="--features cuda"
ENV CUDAARCHS="75;80;86;89;90"
# find_cuda_helper on Linux reads only CUDA_LIBRARY_PATH and joins lib64 onto
# it, so it wants the toolkit root; CUDAToolkit_ROOT feeds find_package.
ENV CUDA_LIBRARY_PATH="/usr/local/cuda"
ENV CUDAToolkit_ROOT="/usr/local/cuda"

FROM base-${VARIANT} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl ca-certificates pkg-config libclang-dev clang \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
# cmake 4.x rejects llama.cpp's older cmake_minimum_required without this.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
WORKDIR /src
COPY . .
RUN cargo build --release ${KNAPPER_CARGO_FEATURES} \
    && cp target/release/knapper /knapper

# Runtime: slim. CUDA links statically (cudart_static, cublas_static,
# cublasLt_static, culibos), so no CUDA userspace is needed here; -lcuda
# resolves at run time against the driver that `--gpus all` injects.
FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /knapper /usr/local/bin/knapper
ENV KNAPPER_HOME=/data
VOLUME ["/data"]
ENTRYPOINT ["knapper"]
