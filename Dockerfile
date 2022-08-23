FROM walkerlab/pytorch:python3.8-torch1.11.0-cuda11.2.1


RUN apt-get update 
RUN apt-get install -y tree fish libnetcdf-dev
RUN pip3 install black flowtorch scikit-image wandb pymc3

RUN git clone https://github.com/sinzlab/insilico-stimuli.git /src/insilico-stimuli &&\
    pip3 install /src/insilico-stimuli

ADD . /src/project