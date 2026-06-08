FROM python:3.10

WORKDIR /data/dpgen2
COPY ./ ./
RUN pip install --no-cache-dir .
