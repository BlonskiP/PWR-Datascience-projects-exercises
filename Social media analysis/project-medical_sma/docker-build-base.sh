#!/bin/bash 

IMG_NAME="sma-base"
TAG="1.0.0"

docker build -t "${IMG_NAME}:${TAG}" -f ./Dockerfile-base .
