FROM walkerlab/pytorch-jupyter:cuda-11.7.1-pytorch-1.13.1-torchvision-0.13.0-torchaudio-0.11.0-ubuntu-20.04

RUN apt-get update 
RUN pip3 install --upgrade pip
RUN pip3 install black scikit-image wandb pymc
RUN pip3 install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

ADD . /src/project