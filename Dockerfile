FROM rayproject/ray:2.0.1-py37

#RUN pip install ray[rllib] torch pettingzoo pytest gym==0.21.0 tqdm pandas
RUN pip install ray[rllib] torch pettingzoo pytest gym==0.21.0 tqdm pandas importlib-metadata==4.13.0
WORKDIR /work
COPY . /work
RUN pip install -e .
