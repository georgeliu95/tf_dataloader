#!/bin/bash
if [ -z "$1" ]
then
    device="all"
else
    device=$1
fi

cd docker
nvidia-docker build -t tf-dataloader:v1 .
cd ..
NV_GPU=${device} \
nvidia-docker run -it --name georgel-tf --rm -v $PWD:/home -u $(id -u):$(id -g) tf-dataloader:v1 /bin/bash