import json
import requests
from typing import Dict, List, Union

import streamlit as st
from spacy_streamlit import visualize_ner
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.tokens import Span

COLORS = ["#ef5350", "#ffee58", "#81c784", "#64b5f6", "#ba68c8", "#b0bec5"]


def write_header() -> None:
    """Write streamlit web app header."""
    st.title("Hugging Face Pipelines Web App")


def text_input() -> str:
    """Generates text input box, that waits for an user input.

    Returns:
        str: User input string.
    """
    text_input = st.text_input("Input a blob of text to be run through the API model.")
    return text_input


def get_prediction(url: str, text: str) -> List[Dict[str, Union[int, float, str]]]:
    """Get prediction by calling the desired API endpoint using the
    user's input.

    Args:
        url (str): API's post url.
        text (str): User input string.

    Returns:
        FinalPrediction: List of dictionaries, each corresponding
            to a final prediction.
    """
    data = json.dumps({"text": text})
    response = requests.post(url, data=data)
    json_response = response.json()
    return json_response


def print_pipeline_info(pipeline_type: str, model: str) -> None:
    """Prints the pipeline type (e.g. Token Classification Pipeline)
    and the model being used (e.g. bert-base).

    Args:
        pipeline_type (str): Type of pipeline.
        model (str): Hugging Face model.
    """
    st.write("## Pipeline Info")
    st.write("**Pipeline Type: **", pipeline_type)
    st.write("**Loaded Model: **", model)


def print_predictions(
    text: str,
    pipeline_type: str,
    predictions: List[Dict[str, Union[int, float, str]]],
    tokens: List[str],
) -> None:
    """Prints each prediction and a prettier version depending on the
    pipeline_type. For the "Token Classification Pipeline" it uses
    spacy_streamlit's visualizer_ner function.

    Args:
        text (str): User input string.
        pipeline_type (str): Type of pipeline.
        predictions (List[Dict[str, Union[int, float, str]]]):  List of
        dictionaries, each corresponding to a final prediction.
    """

    if predictions:
        st.write("## Predictions")

        if pipeline_type == "Token Classification Pipeline":
            # Init spacy's Doc
            doc = Doc(Vocab(strings=tokens), words=tokens)

            # Get ranges mapping
            starts = {}
            ends = {}
            i = -1
            for ix, token in enumerate(tokens):
                start = text.find(token, i + 1)
                i = start
                starts[start] = ix
                ends[start + len(token)] = ix

            # Set entities in spacy's doc, and collect labels
            labels = []
            spans = []
            for prediction in predictions:
                if prediction["start"] not in starts or prediction["end"] not in ends:
                    print("Skipping: ", prediction)
                    continue
                label = prediction["entity_group"]
                if label not in labels:
                    labels.append(label)
                spans.append(
                    Span(
                        doc,
                        start=starts[prediction["start"]],
                        end=ends[prediction["end"]] + 1,
                        label=label,
                    )
                )
            doc.set_ents(spans)

            # Plot using spacy_streamlit
            colors = {k: v for k, v in zip(labels, COLORS[: len(labels)])}
            visualize_ner(doc, labels=labels, show_table=False, colors=colors)

        elif pipeline_type == "Text Classification Pipeline":
            st.write("{} -> Prediction: {}".format(text))

    # Visualize predictions
    if predictions:
        st.write("### Details")
        for v in predictions:
            st.write(v)
    else:
        st.write("No predictions")


def main(url) -> None:
    write_header()
    text = text_input()
    if text:
        response = get_prediction(url, text)
        print_pipeline_info(pipeline_type=response["type"], model=response["model"])
        print_predictions(
            text=text,
            pipeline_type=response["type"],
            predictions=response["predictions"],
            tokens=response["tokens"],
        )


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = "8000"
    url_base = "http://{}:{}/{}/"
    main(url=url_base.format(ip, port, "predict"))
