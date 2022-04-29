FROM dflow:v1.0

WORKDIR /data/dpgen2
ADD requirements.txt ./
RUN pip install -r requirements.txt
COPY ./ ./
RUN pip install .
