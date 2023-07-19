FROM ghcr.io/walkerlab/docker-pytorch-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-20.04

RUN apt-get update 

RUN apt-get install libcudnn8=8.6.0.163-1+cuda11.8

RUN apt install -y fish graphviz

RUN pip3 install --upgrade pip
RUN pip3 install black scikit-image wandb pymc numpyro graphviz
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ADD . /src/project