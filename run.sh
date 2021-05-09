#docker run --gpus all -it --name georgel-tf --rm -v $PWD:/home tensorflow/tensorflow:2.4.1-gpu /bin/bash
nvidia-docker run -it --name georgel-tf --rm -v $PWD:/home tf-dataloader:v1 /bin/bash