FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive 

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Create a conda aloha env
RUN conda create --name aloha python=3.8.10 && \
    echo "source activate aloha" > ~/.bashrc

# Set the PATH for the new environment
ENV PATH /opt/conda/envs/aloha/bin:$PATH

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    vim \ 
    git

RUN /opt/conda/envs/aloha/bin/python -m pip install \
    torchvision \
    torch \
    pyquaternion \
    pyyaml \
    rospkg \
    pexpect \
    mujoco \
    dm_control \
    opencv-python \
    matplotlib \
    einops \
    packaging \
    h5py \
    ipython

# WORKDIR act-plus-plus/detr
COPY detr /detr
WORKDIR /detr
RUN /opt/conda/envs/aloha/bin/python -m pip install -e .

RUN apt-get update && apt-get install -y libglib2.0-0

# Set the working directory
WORKDIR /tempdir

# Uninstall robomimic if it's installed
RUN pip uninstall -y robomimic
RUN git clone https://github.com/ARISE-Initiative/robomimic.git

# Change into the robomimic directory and checkout the desired branch
WORKDIR /tempdir/robomimic
RUN git checkout diffusion-policy-mg

# Install robomimic and diffusers
RUN pip install -e . && pip install diffusers
WORKDIR /act-plus-plus