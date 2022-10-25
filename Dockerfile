FROM rayproject/ray:1.12.1-py38

#RUN pip install ray[rllib] torch pettingzoo pytest gym==0.21.0 tqdm pandas
RUN pip install ray[rllib] torch==1.12.1 pettingzoo pytest gym==0.21.0 tqdm pandas
WORKDIR /work
COPY . /work
RUN pip install -e .
