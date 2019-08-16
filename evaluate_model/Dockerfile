FROM drtools/dask:latest

COPY . /opt/evaluate_model/

USER root

RUN pip install -r /opt/evaluate_model/requirements.txt

WORKDIR /opt/evaluate_model/
