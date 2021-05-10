#!/bin/bash
# docker run --gpus all -it --name georgel-tf --rm -v $PWD:/home tensorflow/tensorflow:2.4.1-gpu /bin/bash
if [ -z "$1" ]
then
    device="all"
else
    device=$1
fi

nvidia-docker run --gpus "device=${device}" -it --name georgel-tf --rm -u $(id -u):$(id -g) -v $PWD:/home tf-dataloader:v1 /bin/bash