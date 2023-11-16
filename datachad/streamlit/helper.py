import os

import deeplake
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.openai_info import get_openai_token_cost_for_model
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from datachad.backend.constants import (
    CHUNK_OVERLAP_PCT,
    CHUNK_SIZE,
    DEFAULT_KNOWLEDGE_BASES,
    DEFAULT_SMART_FAQ,
    DISTANCE_METRIC,
    K_FETCH_K_RATIO,
    MAX_TOKENS,
    MAXIMAL_MARGINAL_RELEVANCE,
    TEMPERATURE,
)
from datachad.backend.deeplake import (
    get_or_create_deeplake_vector_store_display_name,
    get_or_create_deeplake_vector_store_paths_for_user,
)
from datachad.backend.jobs import create_chain, create_vector_store
from datachad.backend.logging import logger
from datachad.backend.models import MODELS, get_tokenizer
from datachad.streamlit.constants import PAGE_ICON

# loads environment variables
load_dotenv()


def init_session_state():
    # Initialise all session state variables with defaults
    SESSION_DEFAULTS = {
        # general usage
        "usage": {},
        "chat_history": StreamlitChatMessageHistory(),
        # authentication
        "openai_api_key": "",
        "activeloop_token": "",
        "activeloop_id": "",
        "credentals": {},
        "auth_ok": False,
        # data upload
        "uploaded_files": None,
        "upload_type": None,
        "upload_name": None,
        # data selection
        "chain": None,
        "knowledge_bases": DEFAULT_KNOWLEDGE_BASES,
        "smart_faq": DEFAULT_SMART_FAQ,
        # advanced options
        "model": MODELS.GPT35TURBO,
        "k_fetch_k_ratio": K_FETCH_K_RATIO,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap_pct": CHUNK_OVERLAP_PCT,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "distance_metric": DISTANCE_METRIC,
        "maximal_marginal_relevance": MAXIMAL_MARGINAL_RELEVANCE,
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def authenticate() -> None:
    # Validate all credentials are set and correct
    # Check for env variables to enable local dev and deployments with shared credentials
    openai_api_key = (
        st.session_state["openai_api_key"]
        or os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY")
    )
    activeloop_token = (
        st.session_state["activeloop_token"]
        or os.environ.get("ACTIVELOOP_TOKEN")
        or st.secrets.get("ACTIVELOOP_TOKEN")
    )
    activeloop_id = (
        st.session_state["activeloop_id"]
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
            openai.models.list()
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
    st.session_state["credentials"] = {
        "openai_api_key": openai_api_key,
        "activeloop_token": activeloop_token,
        "activeloop_id": activeloop_id,
    }
    msg = "Authentification successful!"
    st.session_state["info_container"].info(msg, icon=PAGE_ICON)
    logger.info(msg)


def get_options() -> dict:
    return {
        key: st.session_state[key]
        for key in [
            "model",
            "k_fetch_k_ratio",
            "chunk_size",
            "chunk_overlap_pct",
            "temperature",
            "max_tokens",
            "distance_metric",
            "maximal_marginal_relevance",
        ]
    }


def update_vector_store() -> None:
    try:
        with st.session_state["info_container"], st.spinner("Updating Vector Stores..."):
            options = get_options()
            create_vector_store(
                files=st.session_state["uploaded_files"],
                store_type=st.session_state["upload_type"],
                name=st.session_state["upload_name"],
                options=options,
                credentials=st.session_state["credentials"],
            )
            msg = (
                f"Vector Store built for "
                f"uploaded files: {st.session_state['uploaded_files']} "
                f"and store type: {st.session_state['upload_type']}"
                f"with name: {st.session_state['upload_name']}"
                f"and options: {options}"
            )
            logger.info(msg)
        st.session_state["info_container"].info("Upload successful!", icon=PAGE_ICON)
    except Exception as e:
        msg = f"Failed to build vectore chain with error: {e}"
        logger.error(msg)
        st.session_state["info_container"].error(msg, icon=PAGE_ICON)


def update_chain() -> None:
    try:
        with st.session_state["info_container"], st.spinner("Updating Knowledge Base..."):
            st.session_state["chat_history"].clear()
            options = get_options()
            st.session_state["chain"] = create_chain(
                use_vanilla_llm=st.session_state["use_vanilla_llm"],
                knowledge_bases=st.session_state["knowledge_bases"],
                smart_faq=st.session_state["smart_faq"],
                chat_history=st.session_state["chat_history"],
                options=options,
                credentials=st.session_state["credentials"],
            )
            msg = (
                f"Language chain built for "
                f"knowledge base: {st.session_state['knowledge_bases']} "
                f"and smart faq: {st.session_state['smart_faq']}"
                f"with options: {options}"
            )
            logger.info(msg)
        st.session_state["info_container"].info("Selection successful!", icon=PAGE_ICON)
    except Exception as e:
        msg = f"Failed to build language chain with error: {e}"
        logger.error(msg)
        st.session_state["info_container"].error(msg, icon=PAGE_ICON)


def get_existing_smart_faqs_and_default_index() -> list[str]:
    smart_faqs = get_or_create_deeplake_vector_store_paths_for_user(
        st.session_state["credentials"], "faq"
    )
    index = 0
    if DEFAULT_SMART_FAQ and DEFAULT_SMART_FAQ in smart_faqs:
        # we pick the first smart faq as default
        # so we must sort it to the front
        smart_faqs = set(smart_faqs)
        smart_faqs.remove(DEFAULT_SMART_FAQ)
        smart_faqs = [DEFAULT_SMART_FAQ] + list(smart_faqs)
        index = 1
    # first option should always be None
    smart_faqs = [None] + smart_faqs
    return smart_faqs, index


def get_existing_knowledge_bases() -> list[str]:
    return get_or_create_deeplake_vector_store_paths_for_user(st.session_state["credentials"], "kb")


def format_vector_stores(item: str) -> str:
    if item is not None:
        return get_or_create_deeplake_vector_store_display_name(item)
    return item


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.stream_text = initial_text
        self.chain_state = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.stream_text += token
        self.container.markdown(self.stream_text)

    def on_chain_end(self, outputs, **kwargs) -> None:
        self.chain_state += 1


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs) -> None:
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs) -> None:
        for idx, doc in enumerate(documents):
            try:
                source = os.path.basename(doc.metadata["source"])
                page = doc.metadata.get("page")
                output = f"___\n**Source {idx}:** {source}"
                output += f" (page {page+1})" if page is not None else ""
                self.status.write(output)
            except:
                pass
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


class UsageHandler(BaseCallbackHandler):
    prompt = ""
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    successful_requests = 0
    total_cost = 0

    def update_usage(self) -> None:
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

    def calculate_costs(self) -> None:
        model = st.session_state["model"]
        tokenizer = get_tokenizer({"model": model})
        self.prompt_tokens = len(tokenizer.encode(self.prompt))
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        completion_cost = get_openai_token_cost_for_model(
            model.name, self.completion_tokens, is_completion=True
        )
        prompt_cost = get_openai_token_cost_for_model(model.name, self.prompt_tokens)
        self.total_cost += prompt_cost + completion_cost

    def on_llm_new_token(self, **kwargs) -> None:
        self.completion_tokens += 1

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.successful_requests += 1
        self.prompt += messages[0][0].content

    def on_chain_end(self, outputs, **kwargs) -> None:
        self.calculate_costs()
        self.update_usage()
