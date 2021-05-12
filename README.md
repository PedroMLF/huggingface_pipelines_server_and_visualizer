# Hugging Face Pipelines Server and Visualizer


### Description

The goal of this project is to offer a simple local server and visualization tool around <a href=https://huggingface.co/transformers/main_classes/pipelines.html>Hugging Face Pipelines</a>. It uses [FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/).

---

### Usage

- Create a virtualenv with python3.7+ and run `pip install -r requirements.txt`.
- Modify the `config.yaml` file, by defining the appropriate pipeline and the corresponding model (refer to [Supported Pipelines](#supported-pipelines) for more details), e.g.:

```yaml
pipeline: "TokenClassificationPipeline"
model: "dslim/bert-base-NER"
```

**a) Server + Visualizer:**
- `docker-compose -f docker/py-37_compose.yaml build` .
- `docker-compose -f docker/py-37_compose.yaml up` .
- If necessary modify variables in each service's corresponding `.env` file.


**b) Prediction Only:**
- Build predictions for a specific file using: `python predict.py config.yaml input_file.txt` .
    - Saves all predictions in a json file.
    - Does not require local server.

#### Individual Modules

**a) Server:**
- Launch local server with: `./src/api/start.sh` .

OR

- `docker build -t api -f docker/api-py-37.dockerfile .` .
- `docker container run --name api -p 80:80 api:latest` .

OR

- `docker-compose -f docker/py-37_compose.yaml build api` .
- `docker-compose -f docker/py-37_compose.yaml up api` .

**b) Visualization:**
- Run [streamlit](https://streamlit.io/) visualizer with: `streamlit run src/frontend/run_streamlit.py` .
    - Requires local server to make requests to.

---

### Supported Pipelines

- The following table shows the supported types of pipelines by this project:


| Pipeline                    | Hugging Face Hub Models                                                 |
|:---------------------------:|:-----------------------------------------------------------------------:|
| TextClassificationPipeline  | [Link](https://huggingface.co/models?pipeline_tag=text-classification)  |
| TokenClassificationPipeline | [Link](https://huggingface.co/models?pipeline_tag=token-classification) |

- One type of pipeline can be used to solve different tasks within the corresponding problem class, depending on the model selected from the Hugging Face Hub.

---

### API:

- Implemented with [FastAPI](https://fastapi.tiangolo.com/).
- Follows [tiangolo's uvicorn-gunicorn-docker](https://github.com/tiangolo/uvicorn-gunicorn-docker) structure.
- Supports all the environment variables described [here](https://github.com/tiangolo/uvicorn-gunicorn-docker#environment-variables).
    - To modify them locally do, e.g. `export MAX_WORKERS=2; ./src/api/start.sh` .
    - To modify them when using docker do, e.g. `docker container run --name api -p 80:80 -e MAX_WORKERS="1" api:latest` .
    - To modify them using docker-compose use the docker/api.env file.

---

### Visualizer

- Implemented with [Streamlit](https://streamlit.io/), and with different visualizations according to the type of pipeline.
- If necessary, modify `ip` and `port` to match `src/api/gunicorn_conf.py` by using:
    - `streamlit run src/frontend/run_streamlit.py -- --port=80 --ip=0.0.0.0`
- Example for the `TokenClassificationPipeline`:

<img src="./assets/streamlit_01.png" alt="drawing" width="400"/>
<img src="./assets/streamlit_02.png" alt="drawing" width="400"/>

---

### Other Info

#### Automated Checks

- Run checks for lint, typing, tests, and coverage with `nox` .

#### Linting

- Run `./run_lint.sh` or `black . -l 99` .

#### Typing

- Run `mypy --namespace-packages -p src` .

- Notes:
    - `--namespace-packages`: See [GitHub Issue](https://github.com/python/mypy/issues/1645#issuecomment-472623745) .
    - `-p`: See [GitHub Issue](https://github.com/python/mypy/issues/8944#issuecomment-678725333) .
    - Reused variables cause errors: See [GitHub Issue](https://github.com/python/mypy/issues/1174#issue-129268674) .

#### Tests/Coverage

- Run `coverage run --source=src/ -m pytest` .
- Run `coverage report -m` .
