FROM python:3.7-slim

COPY ./config.yaml /config.yaml
COPY ./docker/api_requirements.txt /requirements.txt
COPY ./src/api /src/api
COPY ./src/pipelines /src/pipelines
COPY ./src/custom_types.py /src/custom_types.py

RUN chmod +x /src/api/start.sh

RUN pip install -r /requirements.txt

ENV PYTHONPATH=/src

EXPOSE 80

CMD ["/src/api/start.sh"]