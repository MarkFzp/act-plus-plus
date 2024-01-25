#!/bin/bash
set -e
set -u

# Define the path to the volume you want to mount
VOLUME_PATH="."

# Build the Docker image. The -t option lets you tag your image so it's easier to find later.
docker build -t aloha .

# Check if the container exists
if [ $(docker ps -a -f name=aloha | grep -w aloha | wc -l) -eq 0 ]; then
    # Create the Docker container
    docker run --name aloha -d -P -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $VOLUME_PATH:/act-plus-plus --ipc=host --pid=host --network=host --gpus=all -w /act-plus-plus aloha:latest
fi

# Start the Docker container
docker start aloha

# Execute command in the Docker container
docker exec -it aloha /bin/bash 