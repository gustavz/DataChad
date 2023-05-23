import logging
import os
import sys

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


def save_uploaded_file() -> str:
    # streamlit uploaded files need to be stored locally
    # before embedded and uploaded to the hub
    uploaded_file = st.session_state["uploaded_file"]
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    file_path = str(DATA_PATH / uploaded_file.name)
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    file = open(file_path, "wb")
    file.write(file_bytes)
    file.close()
    logger.info(f"Saved: {file_path}")
    return file_path


def delete_uploaded_file() -> None:
    # cleanup locally stored files
    file_path = DATA_PATH / st.session_state["uploaded_file"].name
    if os.path.exists(DATA_PATH):
        os.remove(file_path)
        logger.info(f"Removed: {file_path}")
