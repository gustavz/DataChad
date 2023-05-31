import os

import deeplake
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback

from datachad.chain import get_chain
from datachad.constants import (
    ACTIVELOOP_HELP,
    AUTHENTICATION_HELP,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_DATA_SOURCE,
    DISTANCE_METRIC,
    ENABLE_ADVANCED_OPTIONS,
    ENABLE_LOCAL_MODE,
    FETCH_K,
    LOCAL_MODE_DISABLED_HELP,
    MAX_TOKENS,
    MAXIMAL_MARGINAL_RELEVANCE,
    MODE_HELP,
    MODEL_N_CTX,
    OPENAI_HELP,
    PAGE_ICON,
    PROJECT_URL,
    TEMPERATURE,
    K,
)
from datachad.io import delete_files, save_files
from datachad.logging import logger
from datachad.models import MODELS, MODES

# loads environment variables
load_dotenv()


def initialize_session_state():
    # Initialise all session state variables with defaults
    SESSION_DEFAULTS = {
        "past": [],
        "usage": {},
        "chat_history": [],
        "generated": [],
        "auth_ok": False,
        "chain": None,
        "openai_api_key": None,
        "activeloop_token": None,
        "activeloop_org_name": None,
        "uploaded_files": None,
        "info_container": None,
        "data_source": DEFAULT_DATA_SOURCE,
        "mode": MODES.OPENAI,
        "model": MODELS.GPT35TURBO,
        "k": K,
        "fetch_k": FETCH_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "model_n_ctx": MODEL_N_CTX,
        "distance_metric": DISTANCE_METRIC,
        "maximal_marginal_relevance": MAXIMAL_MARGINAL_RELEVANCE,
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def authentication_form() -> None:
    # widget for authentication input form
    st.title("Authentication", help=AUTHENTICATION_HELP)
    with st.form("authentication"):
        openai_api_key = st.text_input(
            f"{st.session_state['mode']} API Key",
            type="password",
            help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )
        activeloop_token = st.text_input(
            "ActiveLoop Token",
            type="password",
            help=ACTIVELOOP_HELP,
            placeholder="Optional, using ours if empty",
        )
        activeloop_org_name = st.text_input(
            "ActiveLoop Organisation Name",
            type="password",
            help=ACTIVELOOP_HELP,
            placeholder="Optional, using ours if empty",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            authenticate(openai_api_key, activeloop_token, activeloop_org_name)


def advanced_options_form() -> None:
    # Input Form that takes advanced options and rebuilds chain with them
    advanced_options = st.checkbox(
        "Advanced Options", help="Caution! This may break things!"
    )
    if advanced_options:
        with st.form("advanced_options"):
            st.selectbox(
                "model",
                options=MODELS.for_mode(st.session_state["mode"]),
                help=f"Learn more about which models are supported [here]({PROJECT_URL})",
                key="model",
            )
            col1, col2 = st.columns(2)
            col1.number_input(
                "temperature",
                min_value=0.0,
                max_value=1.0,
                value=TEMPERATURE,
                help="Controls the randomness of the language model output",
                key="temperature",
            )
            col2.number_input(
                "max_tokens",
                min_value=1,
                max_value=30000,
                value=MAX_TOKENS,
                help="Limits the documents returned from database based on number of tokens",
                key="max_tokens",
            )
            col1.number_input(
                "fetch_k",
                min_value=1,
                max_value=1000,
                value=FETCH_K,
                help="The number of documents to pull from the vector database",
                key="fetch_k",
            )
            col2.number_input(
                "k",
                min_value=1,
                max_value=100,
                value=K,
                help="The number of most similar documents to build the context from",
                key="k",
            )
            col1.number_input(
                "chunk_size",
                min_value=1,
                max_value=100000,
                value=CHUNK_SIZE,
                help=(
                    "The size at which the text is divided into smaller chunks "
                    "before being embedded.\n\nChanging this parameter makes re-embedding "
                    "and re-uploading the data to the database necessary "
                ),
                key="chunk_size",
            )
            col2.number_input(
                "chunk_overlap",
                min_value=0,
                max_value=100000,
                value=CHUNK_OVERLAP,
                help="The size of overlap between splitted document chunks",
                key="chunk_overlap",
            )

            applied = st.form_submit_button("Apply")
            if applied:
                update_chain()


def app_can_be_started():
    # Only start App if authentication is OK or Local Mode
    return st.session_state["auth_ok"] or st.session_state["mode"] == MODES.LOCAL


def update_model_on_mode_change():
    # callback for mode selectbox
    # the default model must be updated for the mode
    st.session_state["model"] = MODELS.for_mode(st.session_state["mode"])[0]
    # Chain needs to be rebuild if app can be started
    if not st.session_state["chain"] is None and app_can_be_started():
        update_chain()


def authentication_and_options_side_bar():
    # Sidebar with Authentication and Advanced Options
    with st.sidebar:
        mode = st.selectbox(
            "Mode",
            MODES.all(),
            key="mode",
            help=MODE_HELP,
            on_change=update_model_on_mode_change,
        )
        if mode == MODES.LOCAL and not ENABLE_LOCAL_MODE:
            st.error(LOCAL_MODE_DISABLED_HELP, icon=PAGE_ICON)
            st.stop()
        if mode != MODES.LOCAL:
            authentication_form()

        st.info(f"Learn how it works [here]({PROJECT_URL})")
        if not app_can_be_started():
            st.stop()

        # Advanced Options
        if ENABLE_ADVANCED_OPTIONS:
            advanced_options_form()


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


def update_chain() -> None:
    # Build chain with parameters from session state and store it back
    # Also delete chat history to not confuse the bot with old context
    try:
        with st.session_state["info_container"], st.spinner("Building Chain..."):
            data_source = st.session_state["data_source"]
            if st.session_state["uploaded_files"] == st.session_state["data_source"]:
                # Save files uploaded by streamlit to disk and set their path as data source.
                # We need to repeat this at every chain update as long as data source is the uploaded file
                # as we need to delete the files after each chain build to make sure to not pollute the app
                # and to ensure data privacy by not storing user data
                data_source = save_files(st.session_state["uploaded_files"])
            st.session_state["chain"] = get_chain(
                data_source=data_source,
                options={
                    "mode": st.session_state["mode"],
                    "model": st.session_state["model"],
                    "k": st.session_state["k"],
                    "fetch_k": st.session_state["fetch_k"],
                    "chunk_size": st.session_state["chunk_size"],
                    "chunk_overlap": st.session_state["chunk_overlap"],
                    "temperature": st.session_state["temperature"],
                    "max_tokens": st.session_state["max_tokens"],
                    "model_n_ctx": st.session_state["model_n_ctx"],
                    "distance_metric": st.session_state["distance_metric"],
                    "maximal_marginal_relevance": st.session_state[
                        "maximal_marginal_relevance"
                    ],
                },
                credentials={
                    "openai_api_key": st.session_state["openai_api_key"],
                    "activeloop_token": st.session_state["activeloop_token"],
                    "activeloop_org_name": st.session_state["activeloop_org_name"],
                },
            )
            if st.session_state["uploaded_files"] == st.session_state["data_source"]:
                # remove uploaded files from disk
                delete_files(st.session_state["uploaded_files"])
            st.session_state["chat_history"] = []
            msg = f"Data source **{st.session_state['data_source']}** is ready to go with model **{st.session_state['model']}**!"
            logger.info(msg)
            st.session_state["info_container"].info(msg, icon=PAGE_ICON)
    except Exception as e:
        msg = f"Failed to build chain for data source **{st.session_state['data_source']}** with model **{st.session_state['model']}**: {e}"
        logger.error(msg)
        st.session_state["info_container"].error(msg, icon=PAGE_ICON)


def update_usage(cb: OpenAICallbackHandler) -> None:
    # Accumulate API call usage via callbacks
    logger.info(f"Usage: {cb}")
    callback_properties = [
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_cost",
    ]
    for prop in callback_properties:
        value = getattr(cb, prop, 0)
        st.session_state["usage"].setdefault(prop, 0)
        st.session_state["usage"][prop] += value


def generate_response(prompt: str) -> str:
    # call the chain to generate responses and add them to the chat history
    with st.spinner("Generating response"), get_openai_callback() as cb:
        response = st.session_state["chain"](
            {"question": prompt, "chat_history": st.session_state["chat_history"]}
        )
        update_usage(cb)
    logger.info(f"Response: '{response}'")
    st.session_state["chat_history"].append((prompt, response["answer"]))
    return response["answer"]
