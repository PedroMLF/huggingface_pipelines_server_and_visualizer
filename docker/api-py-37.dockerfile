FROM python:3.7-slim

COPY ./config.yaml /config.yaml
COPY ./requirements.txt /requirements.txt
COPY ./src/api /src/api
COPY ./src/pipelines /src/pipelines
COPY ./src/custom_types.py /src/custom_types.py

RUN chmod +x /src/api/start.sh

RUN pip install -r /requirements.txt

ENV PYTHONPATH=/src

ARG MAX_WORKERS
ENV MAX_WORKERS="$MAX_WORKERS"

EXPOSE 80

CMD ["/src/api/start.sh"]