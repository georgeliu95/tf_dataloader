#!/bin/bash
cd docker
docker build -t tf-dataloader:v1 .
cd ..
nvidia-docker run -it --name georgel-tf --rm -v $PWD:/home tf-dataloader:v1 /bin/bash