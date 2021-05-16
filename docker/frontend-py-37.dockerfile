FROM python:3.7-slim

COPY ./docker/frontend_requirements.txt /requirements.txt
COPY ./src/frontend /src/frontend
COPY ./src/pipelines /src/pipelines

RUN pip install -r requirements.txt

ENV PYTHONPATH=/src

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/frontend/run_streamlit.py -- --ip ${API_ADDRESS_IP} --port ${API_ADDRESS_PORT}"]