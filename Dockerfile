FROM python:3.11-slim

WORKDIR /data/dpgen2
COPY ./ ./
RUN pip install --no-cache-dir .
