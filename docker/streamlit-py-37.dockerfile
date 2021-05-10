FROM python:3.7-slim

COPY ./requirements.txt /requirements.txt
COPY ./src/frontend /src/frontend
COPY ./src/pipelines /src/pipelines

RUN pip install -r requirements.txt

ENV PYTHONPATH=/src

ARG API_ADDRESS_IP
ARG API_ADDRESS_PORT
ARG STREAMLIT_SERVER_ADDRESS
ENV API_ADDRESS_IP="$API_ADDRESS_IP"
ENV API_ADDRESS_PORT="$API_ADDRESS_PORT"
ENV STREAMLIT_SERVER_ADDRESS="$STREAMLIT_SERVER_ADDRESS"

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/frontend/run_streamlit.py -- --ip ${API_ADDRESS_IP} --port ${API_ADDRESS_PORT}"]