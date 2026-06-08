FROM python:3.11-slim

# git is required at build time: the version is derived from the repository
# by setuptools_scm (see [tool.setuptools_scm] in pyproject.toml).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /data/dpgen2
COPY ./ ./
RUN pip install --no-cache-dir .
