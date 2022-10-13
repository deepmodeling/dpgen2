FROM dptechnology/dflow:latest

WORKDIR /data/dpgen2
COPY ./ ./
RUN pip install .
