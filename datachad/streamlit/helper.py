import os

import deeplake
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from datachad.backend.chain import get_conversational_retrieval_chain
from datachad.backend.deeplake import (
    get_data_source_from_deeplake_dataset_path,
    get_deeplake_vector_store_paths_for_user,
)
from datachad.backend.io import delete_files, save_files
from datachad.backend.logging import logger
from datachad.backend.models import MODELS, MODES
from datachad.streamlit.constants import (
    ACTIVELOOP_HELP,
    APP_NAME,
    AUTHENTICATION_HELP,
    CHUNK_OVERLAP_PCT,
    CHUNK_SIZE,
    DEFAULT_DATA_SOURCE,
    DISTANCE_METRIC,
    ENABLE_ADVANCED_OPTIONS,
    ENABLE_LOCAL_MODE,
    K_FETCH_K_RATIO,
    LOCAL_MODE_DISABLED_HELP,
    MAX_TOKENS,
    MAXIMAL_MARGINAL_RELEVANCE,
    MODE_HELP,
    MODEL_N_CTX,
    OPENAI_HELP,
    PAGE_ICON,
    PROJECT_URL,
    STORE_DOCS_EXTRA,
    TEMPERATURE,
    UPLOAD_HELP,
    USAGE_HELP,
)

# loads environment variables
load_dotenv()


def initialize_session_state():
    # Initialise all session state variables with defaults
    SESSION_DEFAULTS = {
        "usage": {},
        "auth_ok": False,
        "chain": None,
        "openai_api_key": None,
        "activeloop_token": None,
        "activeloop_id": None,
        "uploaded_files": None,
        "info_container": None,
        "data_source": DEFAULT_DATA_SOURCE,
        "mode": MODES.OPENAI,
        "model": MODELS.GPT35TURBO,
        "k_fetch_k_ratio": K_FETCH_K_RATIO,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap_pct": CHUNK_OVERLAP_PCT,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "model_n_ctx": MODEL_N_CTX,
        "distance_metric": DISTANCE_METRIC,
        "maximal_marginal_relevance": MAXIMAL_MARGINAL_RELEVANCE,
        "store_docs_extra": STORE_DOCS_EXTRA,
        "vector_store": None,
        "existing_vector_stores": [],
        "msgs": StreamlitChatMessageHistory(),
        # widget container definition (order matters!)
        "authentication_container": st.sidebar.container(),
        "advanced_options_container": st.sidebar.container(),
        "data_source_container": st.sidebar.container(),
        "info_container": st.sidebar.empty(),
        "usage_container": st.sidebar.container(),
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def page_header():
    # Page options and header
    st.set_option("client.showErrorDetails", True)
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=PAGE_ICON,
        initial_sidebar_state="expanded",
        layout="centered",
    )
    st.markdown(
        f"<h1 style='text-align: center;'>{APP_NAME} {PAGE_ICON} <br> I know all about your data!</h1>",
        unsafe_allow_html=True,
    )


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
        activeloop_id = st.text_input(
            "ActiveLoop Organisation Name",
            type="password",
            help=ACTIVELOOP_HELP,
            placeholder="Optional, using ours if empty",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            authenticate(openai_api_key, activeloop_token, activeloop_id)


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
                help=(
                    "Limits the documents returned from "
                    "database based on number of tokens"
                ),
                key="max_tokens",
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
                max_value=50,
                value=CHUNK_OVERLAP_PCT,
                help="The percentage of overlap between splitted document chunks",
                key="chunk_overlap_pct",
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
    with st.session_state["authentication_container"]:
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

    with st.session_state["advanced_options_container"]:
        # Advanced Options
        if ENABLE_ADVANCED_OPTIONS:
            advanced_options_form()


def authenticate(
    openai_api_key: str, activeloop_token: str, activeloop_id: str
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
    activeloop_id = (
        activeloop_id
        or os.environ.get("ACTIVELOOP_ID")
        or st.secrets.get("ACTIVELOOP_ID")
    )
    if not (openai_api_key and activeloop_token and activeloop_id):
        st.session_state["auth_ok"] = False
        st.error("Credentials neither set nor stored", icon=PAGE_ICON)
        return
    try:
        # Try to access openai and deeplake
        with st.spinner("Authentifying..."):
            openai.api_key = openai_api_key
            openai.Model.list()
            deeplake.exists(
                f"hub://{activeloop_id}/DataChad-Authentication-Check",
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
    st.session_state["activeloop_id"] = activeloop_id
    logger.info("Authentification successful!")


def update_chain() -> None:
    # Build chain with parameters from session state and store it back
    # Also delete chat history to not confuse the bot with old context
    try:
        with st.session_state["info_container"], st.spinner("Building Chain..."):
            vector_store_path = None
            data_source = st.session_state["data_source"]
            if st.session_state["uploaded_files"] == st.session_state["data_source"]:
                # Save files uploaded by streamlit to disk and set their path as data source.
                # We need to repeat this at every chain update as long as data source is the uploaded file
                # as we need to delete the files after each chain build to make sure to not pollute the app
                # and to ensure data privacy by not storing user data
                data_source = save_files(st.session_state["uploaded_files"])
            if st.session_state["vector_store"] == st.session_state["data_source"]:
                # Load an existing vector store if it has been choosen
                vector_store_path = st.session_state["vector_store"]
                data_source = get_data_source_from_deeplake_dataset_path(
                    vector_store_path
                )
            options = {
                "mode": st.session_state["mode"],
                "model": st.session_state["model"],
                "k_fetch_k_ratio": st.session_state["k_fetch_k_ratio"],
                "chunk_size": st.session_state["chunk_size"],
                "chunk_overlap_pct": st.session_state["chunk_overlap_pct"],
                "temperature": st.session_state["temperature"],
                "max_tokens": st.session_state["max_tokens"],
                "model_n_ctx": st.session_state["model_n_ctx"],
                "distance_metric": st.session_state["distance_metric"],
                "maximal_marginal_relevance": st.session_state[
                    "maximal_marginal_relevance"
                ],
                "store_docs_extra": st.session_state["store_docs_extra"],
            }
            credentials = {
                "openai_api_key": st.session_state["openai_api_key"],
                "activeloop_token": st.session_state["activeloop_token"],
                "activeloop_id": st.session_state["activeloop_id"],
            }
            st.session_state["chain"] = get_conversational_retrieval_chain(
                data_source=data_source,
                vector_store_path=vector_store_path,
                options=options,
                credentials=credentials,
                chat_memory=st.session_state["msgs"],
            )
            if st.session_state["uploaded_files"] == st.session_state["data_source"]:
                # remove uploaded files from disk
                delete_files(st.session_state["uploaded_files"])
            # update list of existing vector stores
            st.session_state["existing_vector_stores"] = get_existing_vector_stores(
                options, credentials
            )
            st.session_state["chat_history"] = []
        print("data_source", data_source, type(data_source))
        msg = f"Data source **{data_source}** is ready to go with model **{st.session_state['model']}**!"
        logger.info(msg)
        st.session_state["info_container"].info(msg, icon=PAGE_ICON)
    except Exception as e:
        msg = f"Failed to build chain for data source **{data_source}** with model **{st.session_state['model']}**: {e}"
        logger.error(msg)
        st.session_state["info_container"].error(msg, icon=PAGE_ICON)


def get_existing_vector_stores(options: dict, credentials: dict) -> list[str]:
    return [None] + get_deeplake_vector_store_paths_for_user(options, credentials)


def format_vector_stores(option: str) -> str:
    if option is not None:
        return get_data_source_from_deeplake_dataset_path(option)
    return option


def usage_side_bar():
    # Usage sidebar with total used tokens and costs
    # We put this at the end to be able to show usage after the first response
    with st.session_state["usage_container"]:
        if st.session_state["usage"]:
            st.divider()
            st.title("Usage", help=USAGE_HELP)
            col1, col2 = st.columns(2)
            col1.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
            col2.metric("Total Costs in $", st.session_state["usage"]["total_cost"])


def upload_data_source():
    # file upload and data source input widgets
    uploaded_files = st.session_state["data_source_container"].file_uploader(
        "Upload Files", accept_multiple_files=True, help=UPLOAD_HELP
    )
    data_source = st.session_state["data_source_container"].text_input(
        "Enter any Data Source",
        placeholder="Any path or url pointing to a file or directory of files",
    )

    # generate new chain for new data source / uploaded file
    # make sure to do this only once per input / on change
    if data_source and data_source != st.session_state["data_source"]:
        logger.info(f"Data source provided: '{data_source}'")
        st.session_state["data_source"] = data_source
        update_chain()

    if uploaded_files and uploaded_files != st.session_state["uploaded_files"]:
        logger.info(f"Uploaded files: '{uploaded_files}'")
        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["data_source"] = uploaded_files
        update_chain()

    # we initialize chain after authentication is OK
    # and upload and data source widgets are in place
    # but before existing vector stores are listed
    if st.session_state["chain"] is None:
        update_chain()


def vector_store_selection():
    # List existing vector stores to be able to load them
    vector_store = st.session_state["data_source_container"].selectbox(
        "Select Knowledge Base",
        options=st.session_state["existing_vector_stores"],
        format_func=format_vector_stores,
        index=0,
    )
    if vector_store and vector_store != st.session_state["vector_store"]:
        logger.info(f"Choosen existing vector store: '{vector_store}'")
        st.session_state["vector_store"] = vector_store
        st.session_state["data_source"] = vector_store
        update_chain()


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


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


def chat_interface():
    if len(st.session_state["msgs"].messages) == 0:
        st.session_state["msgs"].clear()
        st.session_state["msgs"].add_ai_message("How can I help you?")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in st.session_state["msgs"].messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"), get_openai_callback() as cb:
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = st.session_state["chain"].run(
                user_query, callbacks=[retrieval_handler, stream_handler]
            )
            logger.info(f"Response: '{response}'")
            update_usage(cb)
