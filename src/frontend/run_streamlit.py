import json
import requests

import streamlit as st


def write_header():
    st.title("Hugging Face Pipeline Web App")


def text_input():
    text_input = st.text_input(
        "Input a blob of text to be run through the API model."
    )
    return text_input


def get_prediction(text: str):
    data = json.dumps({"text": text})
    response = requests.post(url, data=data)
    json_response = response.json()
    return json_response


def main(url):
    write_header()
    text = text_input()
    if text:
        response = get_prediction(text)
        st.write("## Pipeline Info")
        st.write("**Pipeline Type: **", response["type"])
        st.write("**Loaded Model: **", response["model"])
        if response["predictions"]:
            st.write("## Predictions")
        for v in response["predictions"]:
            st.write(v)


if __name__ == "__main__":
    url = "http://127.0.0.1:8000/predict/"
    main(url)
