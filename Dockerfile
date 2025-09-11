##
# cd /scratch/users/bereholschi/mlgen3
# docker build -t tf-torch . 
## or if errors regarding missing uid/gid:
# docker build -t tf-torch --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
# cd /
# docker run -u $(id -u):$(id -g) --gpus all -it -v /scratch/users/bereholschi:/home -w /home -e HOME=/home tf-torch:latest bash
##

## This combination of package versions works well together.

# FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow:2.18.0-gpu

# Create a user with your UID/GID
ARG USER_ID=12907
ARG GROUP_ID=12907
RUN groupadd -g $GROUP_ID usergroup && \
    useradd -m -u $USER_ID -g $GROUP_ID user

# Install ai_edge_torch
RUN pip install --no-cache-dir ai_edge_torch==0.4.0
RUN pip install --no-cache-dir torch==2.7.1
RUN pip install --no-cache-dir torchvision==0.22.1
RUN pip install --no-cache-dir torchaudio==2.7.1

# Install CUDA-enabled JAX
RUN pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set the working directory
WORKDIR /app

# Default command
CMD ["/bin/bash"]


## pip list
# Package                      Version
# ---------------------------- ---------------------
# absl-py                      2.1.0
# ai-edge-litert               1.2.0
# ai-edge-quantizer            0.1.0
# ai-edge-torch                0.4.0
# astunparse                   1.6.3
# blinker                      1.4
# certifi                      2024.8.30
# charset-normalizer           3.4.0
# cryptography                 3.4.8
# dbus-python                  1.2.18
# distro                       1.7.0
# filelock                     3.19.1
# flatbuffers                  24.3.25
# fsspec                       2025.9.0
# gast                         0.6.0
# google-pasta                 0.2.0
# grpcio                       1.67.0
# h5py                         3.12.1
# httplib2                     0.20.2
# idna                         3.10
# immutabledict                4.2.1
# importlib-metadata           4.6.4
# iniconfig                    2.1.0
# jax                          0.7.1
# jax-cuda12-pjrt              0.7.1
# jax-cuda12-plugin            0.7.1
# jaxlib                       0.7.1
# jeepney                      0.7.1
# Jinja2                       3.1.6
# keras                        3.6.0
# keyring                      23.5.0
# launchpadlib                 1.10.16
# lazr.restfulclient           0.14.4
# lazr.uri                     1.0.6
# libclang                     18.1.1
# Markdown                     3.7
# markdown-it-py               3.0.0
# MarkupSafe                   3.0.2
# mdurl                        0.1.2
# ml_dtypes                    0.5.3
# more-itertools               8.10.0
# mpmath                       1.3.0
# namex                        0.0.8
# networkx                     3.5
# numpy                        2.0.2
# nvidia-cublas-cu12           12.6.4.1
# nvidia-cuda-cupti-cu12       12.6.80
# nvidia-cuda-nvcc-cu12        12.9.86
# nvidia-cuda-nvrtc-cu12       12.6.77
# nvidia-cuda-runtime-cu12     12.6.77
# nvidia-cudnn-cu12            9.13.0.50
# nvidia-cufft-cu12            11.3.0.4
# nvidia-cufile-cu12           1.11.1.6
# nvidia-curand-cu12           10.3.7.77
# nvidia-cusolver-cu12         11.7.1.2
# nvidia-cusparse-cu12         12.5.4.2
# nvidia-cusparselt-cu12       0.6.3
# nvidia-nccl-cu12             2.26.2
# nvidia-nvjitlink-cu12        12.6.85
# nvidia-nvshmem-cu12          3.4.5
# nvidia-nvtx-cu12             12.6.77
# oauthlib                     3.2.0
# opt_einsum                   3.4.0
# optree                       0.13.0
# packaging                    24.1
# pillow                       11.3.0
# pip                          24.2
# pluggy                       1.6.0
# protobuf                     5.28.3
# Pygments                     2.18.0
# PyGObject                    3.42.1
# PyJWT                        2.3.0
# pyparsing                    2.4.7
# pytest                       8.4.2
# python-apt                   2.4.0+ubuntu4
# requests                     2.32.3
# rich                         13.9.3
# safetensors                  0.6.2
# scipy                        1.16.1
# SecretStorage                3.3.1
# setuptools                   75.2.0
# six                          1.16.0
# sympy                        1.14.0
# tabulate                     0.9.0
# tensorboard                  2.19.0
# tensorboard-data-server      0.7.2
# tensorflow                   2.19.1
# tensorflow-io-gcs-filesystem 0.37.1
# termcolor                    2.5.0
# torch                        2.7.1
# torch_xla2                   0.0.1.dev202412041639
# torchaudio                   2.7.1
# torchvision                  0.22.1
# triton                       3.3.1
# typing_extensions            4.12.2
# urllib3                      2.2.3
# wadllib                      1.3.6
# Werkzeug                     3.0.5
# wheel                        0.44.0
# wrapt                        1.16.0
# zipp                         1.0.0