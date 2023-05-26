import logging
import os
import re
import shutil
import sys
from typing import List

import deeplake
import openai
import streamlit as st
from dotenv import load_dotenv

from datachad.constants import APP_NAME, DATA_PATH, PAGE_ICON

# loads environment variables
load_dotenv()

logger = logging.getLogger(APP_NAME)


def configure_logger(debug: int = 0) -> None:
    # boilerplate code to enable logging in the streamlit app console
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter("%(message)s")

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False


configure_logger(0)


def authenticate(
    openai_api_key: str, activeloop_token: str, activeloop_org_name: str
) -> None:
    # Validate all credentials are set and correct
    # Check for env variables to enable local dev and deployments with shared credentials
    openai_api_key = (
        openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY")
    )
    activeloop_token = (
        activeloop_token
        or os.environ.get("ACTIVELOOP_TOKEN")
        or st.secrets.get("ACTIVELOOP_TOKEN")
    )
    activeloop_org_name = (
        activeloop_org_name
        or os.environ.get("ACTIVELOOP_ORG_NAME")
        or st.secrets.get("ACTIVELOOP_ORG_NAME")
    )
    if not (openai_api_key and activeloop_token and activeloop_org_name):
        st.session_state["auth_ok"] = False
        st.error("Credentials neither set nor stored", icon=PAGE_ICON)
        return
    try:
        # Try to access openai and deeplake
        with st.spinner("Authentifying..."):
            openai.api_key = openai_api_key
            openai.Model.list()
            deeplake.exists(
                f"hub://{activeloop_org_name}/DataChad-Authentication-Check",
                token=activeloop_token,
            )
    except Exception as e:
        logger.error(f"Authentication failed with {e}")
        st.session_state["auth_ok"] = False
        st.error("Authentication failed", icon=PAGE_ICON)
        return
    # store credentials in the session state
    st.session_state["auth_ok"] = True
    st.session_state["openai_api_key"] = openai_api_key
    st.session_state["activeloop_token"] = activeloop_token
    st.session_state["activeloop_org_name"] = activeloop_org_name
    logger.info("Authentification successful!")


def clean_string_for_storing(string):
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    cleaned_string = re.sub(r"\W+", "-", string)
    cleaned_string = re.sub(r"--+", "- ", cleaned_string).strip("-")
    return cleaned_string


def concatenate_file_names(strings: List[str], n_max: int = 30) -> str:
    # Calculate N based on the length of the list
    n = max(1, n_max // len(strings))
    result = ""
    # Add up the first N characters of each string
    for string in sorted(strings):
        result += f"-{string[:n]}"
    return clean_string_for_storing(result)


def get_data_source_path(uploaded_files):
    if len(uploaded_files) > 1:
        # we create a folder name by adding up parts of the file names
        path = DATA_PATH / concatenate_file_names([f.name for f in uploaded_files])
    else:
        path = DATA_PATH / uploaded_files[0].name
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_file(file, path):
    file_path = str(path / file.name)
    file.seek(0)
    file_bytes = file.read()
    file = open(file_path, "wb")
    file.write(file_bytes)
    file.close()
    logger.info(f"Saved: {file_path}")


def save_uploaded_files() -> str:
    # streamlit uploaded files need to be stored locally
    # before embedded and uploaded to the hub
    uploaded_files = st.session_state["uploaded_files"]
    data_source_path = get_data_source_path(uploaded_files)
    logger.info(f"{data_source_path=}")
    for file in uploaded_files:
        save_file(file, data_source_path)
    return str(data_source_path)


def delete_uploaded_files() -> None:
    # cleanup locally stored files
    data_source_path = get_data_source_path(st.session_state["uploaded_files"])
    if os.path.isdir(data_source_path):
        shutil.rmtree(data_source_path)
    elif os.path.isfile(data_source_path):
        os.remove(data_source_path)
    else:
        return
    logger.info(f"Removed: {data_source_path}")
