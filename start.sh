#!/bin/bash
cd docker
docker build -t tf-dataloader:v1 .
cd ..
usr_id=`id | grep -oE "uid=[0-9]+" | grep -oE "[0-9]+"`
nvidia-docker run -it --name georgel-tf --rm -v $PWD:/home --user ${usr_id} tf-dataloader:v1 /bin/bash