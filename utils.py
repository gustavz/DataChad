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
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake, VectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from constants import (
    APP_NAME,
    CHUNK_SIZE,
    DATA_PATH,
    FETCH_K,
    MAX_TOKENS,
    MODEL,
    PAGE_ICON,
    REPO_URL,
    TEMPERATURE,
    K,
)

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


def advanced_options_form() -> None:
    # Input Form that takes advanced options and rebuilds chain with them
    advanced_options = st.checkbox(
        "Advanced Options", help="Caution! This may break things!"
    )
    if advanced_options:
        with st.form("advanced_options"):
            temperature = st.slider(
                "temperature",
                min_value=0.0,
                max_value=1.0,
                value=TEMPERATURE,
                help="Controls the randomness of the language model output",
            )
            col1, col2 = st.columns(2)
            fetch_k = col1.number_input(
                "k_fetch",
                min_value=1,
                max_value=1000,
                value=FETCH_K,
                help="The number of documents to pull from the vector database",
            )
            k = col2.number_input(
                "k",
                min_value=1,
                max_value=100,
                value=K,
                help="The number of most similar documents to build the context from",
            )
            chunk_size = col1.number_input(
                "chunk_size",
                min_value=1,
                max_value=100000,
                value=CHUNK_SIZE,
                help=(
                    "The size at which the text is divided into smaller chunks "
                    "before being embedded.\n\nChanging this parameter makes re-embedding "
                    "and re-uploading the data to the database necessary "
                ),
            )
            max_tokens = col2.number_input(
                "max_tokens",
                min_value=1,
                max_value=4069,
                value=MAX_TOKENS,
                help="Limits the documents returned from database based on number of tokens",
            )
            applied = st.form_submit_button("Apply")
            if applied:
                st.session_state["k"] = k
                st.session_state["fetch_k"] = fetch_k
                st.session_state["chunk_size"] = chunk_size
                st.session_state["temperature"] = temperature
                st.session_state["max_tokens"] = max_tokens
                update_chain()


def save_uploaded_file(uploaded_file: UploadedFile) -> str:
    # streamlit uploaded files need to be stored locally
    # before embedded and uploaded to the hub
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


def delete_uploaded_file(uploaded_file: UploadedFile) -> None:
    # cleanup locally stored files
    file_path = DATA_PATH / uploaded_file.name
    if os.path.exists(DATA_PATH):
        os.remove(file_path)
        logger.info(f"Removed: {file_path}")


def handle_load_error(e: str = None) -> None:
    error_msg = f"Failed to load '{st.session_state['data_source']}':\n\n{e}"
    st.error(error_msg, icon=PAGE_ICON)
    logger.error(error_msg)
    st.stop()


def load_git(data_source: str, chunk_size: int = CHUNK_SIZE) -> List[Document]:
    # We need to try both common main branches
    # Thank you github for the "master" to "main" switch
    # we need to make sure the data path exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    repo_name = data_source.split("/")[-1].split(".")[0]
    repo_path = str(DATA_PATH / repo_name)
    clone_url = data_source
    if os.path.exists(repo_path):
        clone_url = None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )
    branches = ["main", "master"]
    for branch in branches:
        try:
            docs = GitLoader(repo_path, clone_url, branch).load_and_split(text_splitter)
            break
        except Exception as e:
            logger.error(f"Error loading git: {e}")
    if os.path.exists(repo_path):
        # cleanup repo afterwards
        shutil.rmtree(repo_path)
    try:
        return docs
    except:
        msg = "Make sure to use HTTPS git repo links"
        handle_load_error(msg)


def load_any_data_source(
    data_source: str, chunk_size: int = CHUNK_SIZE
) -> List[Document]:
    # Ugly thing that decides how to load data
    # It aint much, but it's honest work
    is_text = data_source.endswith(".txt")
    is_web = data_source.startswith("http")
    is_pdf = data_source.endswith(".pdf")
    is_csv = data_source.endswith("csv")
    is_html = data_source.endswith(".html")
    is_git = data_source.endswith(".git")
    is_notebook = data_source.endswith(".ipynb")
    is_doc = data_source.endswith(".doc")
    is_py = data_source.endswith(".py")
    is_dir = os.path.isdir(data_source)
    is_file = os.path.isfile(data_source)

    loader = None
    if is_dir:
        loader = DirectoryLoader(data_source, recursive=True, silent_errors=True)
    elif is_git:
        return load_git(data_source, chunk_size)
    elif is_web:
        if is_pdf:
            loader = OnlinePDFLoader(data_source)
        else:
            loader = WebBaseLoader(data_source)
    elif is_file:
        if is_text:
            loader = TextLoader(data_source)
        elif is_notebook:
            loader = NotebookLoader(data_source)
        elif is_pdf:
            loader = UnstructuredPDFLoader(data_source)
        elif is_html:
            loader = UnstructuredHTMLLoader(data_source)
        elif is_doc:
            loader = UnstructuredWordDocumentLoader(data_source)
        elif is_csv:
            loader = CSVLoader(data_source, encoding="utf-8")
        elif is_py:
            loader = PythonLoader(data_source)
        else:
            loader = UnstructuredFileLoader(data_source)
    try:
        # Chunk size is a major trade-off parameter to control result accuracy over computaion
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        )
        docs = loader.load_and_split(text_splitter)
        logger.info(f"Loaded: {len(docs)} document chucks")
        return docs
    except Exception as e:
        msg = (
            e
            if loader
            else f"No Loader found for your data source. Consider contributing:  {REPO_URL}!"
        )
        handle_load_error(msg)


def clean_data_source_string(data_source_string: str) -> str:
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    dashed_string = re.sub(r"\W+", "-", data_source_string)
    cleaned_string = re.sub(r"--+", "- ", dashed_string).strip("-")
    return cleaned_string


def setup_vector_store(data_source: str, chunk_size: int = CHUNK_SIZE) -> VectorStore:
    # either load existing vector store or upload a new one to the hub
    embeddings = OpenAIEmbeddings(
        disallowed_special=(), openai_api_key=st.session_state["openai_api_key"]
    )
    data_source_name = clean_data_source_string(data_source)
    dataset_path = f"hub://{st.session_state['activeloop_org_name']}/{data_source_name}-{chunk_size}"
    if deeplake.exists(dataset_path, token=st.session_state["activeloop_token"]):
        with st.spinner("Loading vector store..."):
            logger.info(f"Dataset '{dataset_path}' exists -> loading")
            vector_store = DeepLake(
                dataset_path=dataset_path,
                read_only=True,
                embedding_function=embeddings,
                token=st.session_state["activeloop_token"],
            )
    else:
        with st.spinner("Reading, embedding and uploading data to hub..."):
            logger.info(f"Dataset '{dataset_path}' does not exist -> uploading")
            docs = load_any_data_source(data_source, chunk_size)
            vector_store = DeepLake.from_documents(
                docs,
                embeddings,
                dataset_path=dataset_path,
                token=st.session_state["activeloop_token"],
            )
    return vector_store


def build_chain(
    data_source: str,
    k: int = K,
    fetch_k: int = FETCH_K,
    chunk_size: int = CHUNK_SIZE,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> ConversationalRetrievalChain:
    # create the langchain that will be called to generate responses
    vector_store = setup_vector_store(data_source, chunk_size)
    retriever = vector_store.as_retriever()
    # Search params "fetch_k" and "k" define how many documents are pulled from the hub
    # and selected after the document matching to build the context
    # that is fed to the model together with your prompt
    search_kwargs = {
        "maximal_marginal_relevance": True,
        "distance_metric": "cos",
        "fetch_k": fetch_k,
        "k": k,
    }
    retriever.search_kwargs.update(search_kwargs)
    model = ChatOpenAI(
        model_name=MODEL,
        temperature=temperature,
        openai_api_key=st.session_state["openai_api_key"],
    )
    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        # we limit the maximum number of used tokens
        # to prevent running into the models token limit of 4096
        max_tokens_limit=max_tokens,
    )
    logger.info(f"Data source '{data_source}' is ready to go!")
    return chain


def update_chain() -> None:
    # Build chain with parameters from session state and store it back
    # Also delete chat history to not confuse the bot with old context
    try:
        st.session_state["chain"] = build_chain(
            data_source=st.session_state["data_source"],
            k=st.session_state["k"],
            fetch_k=st.session_state["fetch_k"],
            chunk_size=st.session_state["chunk_size"],
            temperature=st.session_state["temperature"],
            max_tokens=st.session_state["max_tokens"],
        )
        st.session_state["chat_history"] = []
    except Exception as e:
        msg = f"Failed to build chain for data source '{st.session_state['data_source']}' with error: {e}"
        logger.error(msg)
        st.error(msg, icon=PAGE_ICON)


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
