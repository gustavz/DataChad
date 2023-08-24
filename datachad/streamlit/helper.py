import os

import deeplake
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.openai_info import get_openai_token_cost_for_model
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from datachad.backend.chain import get_conversational_retrieval_chain
from datachad.backend.constants import (
    CHUNK_OVERLAP_PCT,
    CHUNK_SIZE,
    DISTANCE_METRIC,
    K_FETCH_K_RATIO,
    MAX_TOKENS,
    MAXIMAL_MARGINAL_RELEVANCE,
    STORE_DOCS_EXTRA,
    TEMPERATURE,
)
from datachad.backend.deeplake import (
    get_data_source_from_deeplake_dataset_path,
    get_deeplake_vector_store_paths_for_user,
)
from datachad.backend.io import delete_files, save_files
from datachad.backend.logging import logger
from datachad.backend.models import MODELS, get_tokenizer
from datachad.streamlit.constants import DEFAULT_DATA_SOURCE, PAGE_ICON

# loads environment variables
load_dotenv()


def init_session_state():
    # Initialise all session state variables with defaults
    SESSION_DEFAULTS = {
        "edited": None,
        "usage": {},
        "auth_ok": False,
        "chain": None,
        "openai_api_key": None,
        "activeloop_token": None,
        "activeloop_id": None,
        "uploaded_files": None,
        "info_container": None,
        "data_source": DEFAULT_DATA_SOURCE,
        "model": MODELS.GPT35TURBO,
        "k_fetch_k_ratio": K_FETCH_K_RATIO,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap_pct": CHUNK_OVERLAP_PCT,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "distance_metric": DISTANCE_METRIC,
        "maximal_marginal_relevance": MAXIMAL_MARGINAL_RELEVANCE,
        "store_docs_extra": STORE_DOCS_EXTRA,
        "vector_store": None,
        "existing_vector_stores": [],
        "chat_history": StreamlitChatMessageHistory(),
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
        with st.session_state["info_container"], st.spinner("Authentifying..."):
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
        with st.session_state["info_container"], st.spinner(
            "Loading Knowledge Base..."
        ):
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
                "model": st.session_state["model"],
                "k_fetch_k_ratio": st.session_state["k_fetch_k_ratio"],
                "chunk_size": st.session_state["chunk_size"],
                "chunk_overlap_pct": st.session_state["chunk_overlap_pct"],
                "temperature": st.session_state["temperature"],
                "max_tokens": st.session_state["max_tokens"],
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
                chat_memory=st.session_state["chat_history"],
            )
            if st.session_state["uploaded_files"] == st.session_state["data_source"]:
                # remove uploaded files from disk
                delete_files(st.session_state["uploaded_files"])
            # update list of existing vector stores
            st.session_state["existing_vector_stores"] = get_existing_vector_stores(
                credentials
            )
            st.session_state["chat_history"].clear()
        print("data_source", data_source, type(data_source))
        msg = f"Data source **{data_source}** is ready to go with model **{st.session_state['model']}**!"
        logger.info(msg)
        st.session_state["info_container"].info(msg, icon=PAGE_ICON)
    except Exception as e:
        msg = f"Failed to build chain for data source **{data_source}** with model **{st.session_state['model']}**: {e}"
        logger.error(msg)
        st.session_state["info_container"].error(msg, icon=PAGE_ICON)


def get_existing_vector_stores(credentials: dict) -> list[str]:
    return [None] + get_deeplake_vector_store_paths_for_user(credentials)


def format_vector_stores(item: str) -> str:
    if item is not None:
        return get_data_source_from_deeplake_dataset_path(item)
    return item


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.chain_state = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # we don't want to write the condense_question_prompt response
        if self.chain_state > 0:
            self.text += token
            self.container.markdown(self.text)

    def on_chain_end(self, outputs, **kwargs):
        self.chain_state += 1


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            page = doc.metadata.get("page")
            output = f"___\n**Source {idx}:** {source}"
            output += f" (page {page}" if page else ""
            self.status.write(output)
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


class UsageHandler(BaseCallbackHandler):
    prompt = ""
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    successful_requests = 0
    total_cost = 0

    def update_usage(self):
        usage_properties = [
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "successful_requests",
            "total_cost",
        ]
        for prop in usage_properties:
            value = getattr(self, prop, 0)
            setattr(self, prop, 0)
            st.session_state["usage"].setdefault(prop, 0)
            st.session_state["usage"][prop] += value

    def calculate_costs(self):
        model = st.session_state["model"]
        tokenizer = get_tokenizer({"model": model})
        self.prompt_tokens = len(tokenizer.encode(self.prompt))
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        completion_cost = get_openai_token_cost_for_model(
            model.name, self.completion_tokens, is_completion=True
        )
        prompt_cost = get_openai_token_cost_for_model(model.name, self.prompt_tokens)
        self.total_cost += prompt_cost + completion_cost

    def on_llm_new_token(self, **kwargs):
        self.completion_tokens += 1

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.successful_requests += 1
        self.prompt += messages[0][0].content

    def on_chain_end(self, outputs, **kwargs):
        self.calculate_costs()
        self.update_usage()
