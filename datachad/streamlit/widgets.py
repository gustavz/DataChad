import streamlit as st

from datachad.backend.constants import (
    CHUNK_OVERLAP_PCT,
    CHUNK_SIZE,
    DEFAULT_KNOWLEDGE_BASES,
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
    get_existing_knowledge_bases,
    get_existing_smart_faqs_and_default_index,
    update_chain,
    update_vector_store,
)


def page_header():
    # Page options and header
    st.set_option("client.showErrorDetails", True)
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=PAGE_ICON,
        initial_sidebar_state="expanded",
        layout="wide",
    )
    st.markdown(
        f"<h1 style='text-align: center;'>{APP_NAME} {PAGE_ICON} <br> I know all about your data!</h1>",
        unsafe_allow_html=True,
    )


def init_widgets():
    # widget container definition (order matters!)
    with st.sidebar:
        st.session_state["authentication_container"] = st.container()
        st.session_state["data_upload_container"] = st.container()
        st.session_state["data_selection_container"] = st.container()
        st.session_state["usage_container"] = st.container()
        st.session_state["info_container"] = st.empty()


def authentication_widget():
    # Sidebar with Authentication
    with st.session_state["authentication_container"]:
        st.info(f"Learn how it works [here]({PROJECT_URL})")
        with st.expander("Authentication", expanded=not st.session_state["auth_ok"]), st.form(
            "authentication"
        ):
            st.text_input(
                f"OpenAI API Key",
                type="password",
                help=OPENAI_HELP,
                placeholder="This field is mandatory",
                key="openai_api_key",
            )
            st.text_input(
                "ActiveLoop Token",
                type="password",
                help=ACTIVELOOP_HELP,
                placeholder="Optional, using ours if empty",
                key="activeloop_token",
            )
            st.text_input(
                "ActiveLoop Organisation Name",
                type="password",
                help=ACTIVELOOP_HELP,
                placeholder="Optional, using ours if empty",
                key="activeloop_id",
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                authenticate()

    if not st.session_state["auth_ok"]:
        st.stop()


def upload_options_widget():
    if ENABLE_ADVANCED_OPTIONS:
        col1, col2 = st.columns(2)
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


def selection_options_widget():
    if ENABLE_ADVANCED_OPTIONS:
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
            help=("Limits the documents returned from " "database based on number of tokens"),
            key="max_tokens",
        )


def data_upload_widget():
    with st.session_state["data_upload_container"], st.expander("Data Upload"), st.form(
        "data_upload"
    ):
        st.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
            help=UPLOAD_HELP,
            key="uploaded_files",
        )
        st.radio("Upload Type", options=["Knowledge Base", "Smart FAQ"], key="upload_type")
        st.text_input(
            "Upload Name",
            placeholder="Give a descriptive and unique name",
            key="upload_name",
        )
        upload_options_widget()
        submitted = st.form_submit_button("Submit")
    if submitted:
        if st.session_state["uploaded_files"] and st.session_state["upload_name"]:
            update_vector_store()
        else:
            st.session_state["info_container"].error(
                "Missing required files and name!", icon=PAGE_ICON
            )


def data_selection_widget():
    with st.session_state["data_selection_container"], st.expander("Data Selection"), st.form(
        "data_selection"
    ):
        existing_smart_faqs, default_index = get_existing_smart_faqs_and_default_index()
        st.selectbox(
            "Select a single Smart FAQ",
            options=existing_smart_faqs,
            format_func=format_vector_stores,
            # TODO: chose and store default smart FAQ
            index=default_index,
            key="smart_faq",
        )
        existing_knowledge_bases = get_existing_knowledge_bases()
        st.multiselect(
            "Select multiple Knowledge Bases",
            options=existing_knowledge_bases,
            format_func=format_vector_stores,
            # TODO: chose and store default knowledge bases
            default=DEFAULT_KNOWLEDGE_BASES,
            key="knowledge_bases",
        )
        selection_options_widget()
        submitted = st.form_submit_button("Submit")
    if submitted:
        update_chain()
    if not st.session_state["chain"]:
        update_chain()


def chat_interface_widget():
    if len(st.session_state["chat_history"].messages) == 0:
        st.session_state["chat_history"].clear()

    st.chat_message("assistant").write("How can I help you?")

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
            col2.metric("Total Costs in $", round(st.session_state["usage"]["total_cost"], 4))
