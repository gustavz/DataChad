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
from langchain.base_language import BaseLanguageModel
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    EverNoteLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PDFMinerLoader,
    PythonLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings.openai import Embeddings, OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake, VectorStore
from langchain.llms import GPT4All, LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings

from constants import APP_NAME, DATA_PATH, PAGE_ICON, PROJECT_URL, LLAMACPP_MODEL_PATH, GPT4ALL_MODEL_PATH

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


class AutoGitLoader:
    def __init__(self, data_source: str) -> None:
        self.data_source = data_source

    def load(self) -> List[Document]:
        # We need to try both common main branches
        # Thank you github for the "master" to "main" switch
        # we need to make sure the data path exists
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        repo_name = self.data_source.split("/")[-1].split(".")[0]
        repo_path = str(DATA_PATH / repo_name)
        clone_url = self.data_source
        if os.path.exists(repo_path):
            clone_url = None
        branches = ["main", "master"]
        for branch in branches:
            try:
                docs = GitLoader(repo_path, clone_url, branch).load()
                break
            except Exception as e:
                logger.error(f"Error loading git: {e}")
        if os.path.exists(repo_path):
            # cleanup repo afterwards
            shutil.rmtree(repo_path)
        try:
            return docs
        except:
            raise RuntimeError("Make sure to use HTTPS GitHub repo links")


FILE_LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf-8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".ipynb": (NotebookLoader, {}),
    ".py": (PythonLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}

WEB_LOADER_MAPPING = {
    ".git": (AutoGitLoader, {}),
    ".pdf": (OnlinePDFLoader, {}),
}


def get_loader(file_path: str, mapping: dict, default_loader: BaseLoader) -> BaseLoader:
    # Choose loader from mapping, load default if no match found
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in mapping:
        loader_class, loader_args = mapping[ext]
        loader = loader_class(file_path, **loader_args)
    else:
        loader = default_loader(file_path)
    return loader


def load_data_source() -> List[Document]:
    # Ugly thing that decides how to load data
    # It aint much, but it's honest work
    data_source = st.session_state["data_source"]
    is_web = data_source.startswith("http")
    is_dir = os.path.isdir(data_source)
    is_file = os.path.isfile(data_source)

    loader = None
    if is_dir:
        loader = DirectoryLoader(data_source, recursive=True, silent_errors=True)
    elif is_web:
        loader = get_loader(data_source, WEB_LOADER_MAPPING, WebBaseLoader)
    elif is_file:
        loader = get_loader(data_source, FILE_LOADER_MAPPING, UnstructuredFileLoader)
    try:
        # Chunk size is a major trade-off parameter to control result accuracy over computaion
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state["chunk_size"],
            chunk_overlap=st.session_state["chunk_overlap"],
        )
        docs = loader.load()
        docs = text_splitter.split_documents(docs)
        logger.info(f"Loaded: {len(docs)} document chucks")
        return docs
    except Exception as e:
        msg = (
            e
            if loader
            else f"No Loader found for your data source. Consider contributing:  {PROJECT_URL}!"
        )
        error_msg = f"Failed to load '{st.session_state['data_source']}':\n\n{msg}"
        st.error(error_msg, icon=PAGE_ICON)
        logger.error(error_msg)
        st.stop()


def get_dataset_name() -> str:
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    dashed_string = re.sub(r"\W+", "-", st.session_state["data_source"])
    cleaned_string = re.sub(r"--+", "- ", dashed_string).strip("-")
    return cleaned_string


def get_model() -> BaseLanguageModel:
    match st.session_state["model"]:
        case "gpt-3.5-turbo":
            model = ChatOpenAI(
                model_name=st.session_state["model"],
                temperature=st.session_state["temperature"],
                openai_api_key=st.session_state["openai_api_key"],
            )
        case "LlamaCpp":
            model = LlamaCpp(
                model_path=LLAMACPP_MODEL_PATH,
                n_ctx=st.session_state["model_n_ctx"],
                temperature=st.session_state["temperature"],
                verbose=True,
            )
        case "GPT4All":
            model = GPT4All(
                model=GPT4ALL_MODEL_PATH,
                n_ctx=st.session_state["model_n_ctx"],
                backend="gptj",
                temp=st.session_state["temperature"],
                verbose=True,
            )
        # Add more models as needed
        case _default:
            msg = f"Model {st.session_state['model']} not supported!"
            logger.error(msg)
            st.error(msg)
            exit
    return model


def get_embeddings() -> Embeddings:
    match st.session_state["embeddings"]:
        case "openai":
            embeddings = OpenAIEmbeddings(
                disallowed_special=(), openai_api_key=st.session_state["openai_api_key"]
            )
        case "huggingface-Fall-MiniLM-L6-v2":
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Add more embeddings as needed
        case _default:
            msg = f"Embeddings {st.session_state['embeddings']} not supported!"
            logger.error(msg)
            st.error(msg)
            exit
    return embeddings


def get_vector_store() -> VectorStore:
    # either load existing vector store or upload a new one to the hub
    embeddings = get_embeddings()
    dataset_name = get_dataset_name()
    dataset_path = f"hub://{st.session_state['activeloop_org_name']}/{dataset_name}-{st.session_state['chunk_size']}"
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
            docs = load_data_source()
            vector_store = DeepLake.from_documents(
                docs,
                embeddings,
                dataset_path=dataset_path,
                token=st.session_state["activeloop_token"],
            )
    return vector_store


def get_chain() -> ConversationalRetrievalChain:
    # create the langchain that will be called to generate responses
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()
    # Search params "fetch_k" and "k" define how many documents are pulled from the hub
    # and selected after the document matching to build the context
    # that is fed to the model together with your prompt
    search_kwargs = {
        "maximal_marginal_relevance": True,
        "distance_metric": "cos",
        "fetch_k": st.session_state["fetch_k"],
        "k": st.session_state["k"],
    }
    retriever.search_kwargs.update(search_kwargs)
    model = get_model()
    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        # we limit the maximum number of used tokens
        # to prevent running into the models token limit of 4096
        max_tokens_limit=st.session_state["max_tokens"],
    )
    return chain


def update_chain() -> None:
    # Build chain with parameters from session state and store it back
    # Also delete chat history to not confuse the bot with old context
    try:
        st.session_state["chain"] = get_chain()
        st.session_state["chat_history"] = []
        msg = f"Data source '{st.session_state['data_source']}' is ready to go!"
        logger.info(msg)
        st.info(msg, icon=PAGE_ICON)
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
