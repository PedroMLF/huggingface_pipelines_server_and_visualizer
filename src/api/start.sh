#! /usr/bin/env sh
set -e

GUNICORN_CONF_PATH=src/api/gunicorn_conf.py
API_MODULE=src.api.api:app

exec gunicorn -k uvicorn.workers.UvicornWorker -c "$GUNICORN_CONF_PATH" "$API_MODULE"