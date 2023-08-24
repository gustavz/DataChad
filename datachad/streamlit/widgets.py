import streamlit as st

from datachad.backend.constants import (
    CHUNK_OVERLAP_PCT,
    CHUNK_SIZE,
    ENABLE_ADVANCED_OPTIONS,
    MAX_TOKENS,
    TEMPERATURE,
)
from datachad.backend.logging import logger
from datachad.backend.models import MODELS
from datachad.streamlit.constants import (
    ACTIVELOOP_HELP,
    APP_NAME,
    OPENAI_HELP,
    PAGE_ICON,
    PROJECT_URL,
    UPLOAD_HELP,
)
from datachad.streamlit.helper import (
    PrintRetrievalHandler,
    StreamHandler,
    UsageHandler,
    authenticate,
    format_vector_stores,
    update_chain,
)


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


def init_widgets():
    # widget container definition (order matters!)
    with st.sidebar:
        st.session_state["authentication_container"] = st.container()
        st.session_state["data_source_container"] = st.container()
        st.session_state["advanced_options_container"] = st.container()
        st.session_state["usage_container"] = st.container()
        st.session_state["info_container"] = st.empty()


def advanced_options_widget() -> None:
    # Input Form that takes advanced options and rebuilds chain with them
    if ENABLE_ADVANCED_OPTIONS:
        with st.session_state["advanced_options_container"], st.expander(
            "Advanced Options"
        ), st.form("advanced_options"):
            st.selectbox(
                "model",
                options=MODELS.all(),
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


def authentication_widget():
    # Sidebar with Authentication
    with st.session_state["authentication_container"]:
        st.info(f"Learn how it works [here]({PROJECT_URL})")
        with st.expander(
            "Authentication", expanded=not st.session_state["auth_ok"]
        ), st.form("authentication"):
            openai_api_key = st.text_input(
                f"OpenAI API Key",
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
    if not st.session_state["auth_ok"]:
        st.stop()


def select_data_source_widget():
    # file upload and data source input widgets
    with st.session_state["data_source_container"], st.expander("Data Source"):
        uploaded_files = st.file_uploader(
            "Upload Files", accept_multiple_files=True, help=UPLOAD_HELP
        )
        data_source = st.text_input(
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

        # List existing vector stores to be able to load them
        vector_store = st.selectbox(
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


def chat_interface_widget():
    if len(st.session_state["chat_history"].messages) == 0:
        st.session_state["chat_history"].clear()
        st.session_state["chat_history"].add_ai_message("How can I help you?")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in st.session_state["chat_history"].messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            usage_handler = UsageHandler()
            response = st.session_state["chain"].run(
                user_query, callbacks=[retrieval_handler, stream_handler, usage_handler]
            )
            logger.info(f"Response: '{response}'")


def usage_widget():
    # Usage sidebar with total used tokens and costs
    # We put this at the end to be able to show usage after the first response
    if st.session_state["usage"]:
        with st.session_state["usage_container"], st.expander("Usage"):
            col1, col2 = st.columns(2)
            col1.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
            col2.metric(
                "Total Costs in $", round(st.session_state["usage"]["total_cost"], 4)
            )
