import os
import re

import logging
import sys
import openai
import deeplake
import shutil
import streamlit as st
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake

from constants import DATA_PATH, MODEL, PAGE_ICON, APP_NAME


logger = logging.getLogger(APP_NAME)


def configure_logger(debug=0):
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter("%(message)s")

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False


configure_logger(0)


def validate_keys(openai_key, activeloop_token, activeloop_org_name):
    # Validate all API related variables are set and correct
    all_keys = [openai_key, activeloop_token, activeloop_org_name]
    if any(all_keys):
        if not all(all_keys):
            st.session_state["auth_ok"] = False
            st.error("You need to fill all fields", icon=PAGE_ICON)
            st.stop()
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
        os.environ["ACTIVELOOP_ORG_NAME"] = activeloop_org_name
    else:
        # Bypass for local development or deployments with stored credentials
        # either env variables or streamlit secrets need to be set
        try:
            try:
                assert os.environ.get("OPENAI_API_KEY")
                assert os.environ.get("ACTIVELOOP_TOKEN")
                assert os.environ.get("ACTIVELOOP_ORG_NAME")
            except:
                assert st.secrets.get("OPENAI_API_KEY")
                assert st.secrets.get("ACTIVELOOP_TOKEN")
                assert st.secrets.get("ACTIVELOOP_ORG_NAME")

                os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY")
                os.environ["ACTIVELOOP_TOKEN"] = st.secrets.get("ACTIVELOOP_TOKEN")
                os.environ["ACTIVELOOP_ORG_NAME"] = st.secrets.get(
                    "ACTIVELOOP_ORG_NAME"
                )
        except:
            st.session_state["auth_ok"] = False
            st.error("No credentials stored and nothing submitted", icon=PAGE_ICON)
            st.stop()
    try:
        # Try to access openai and deeplake
        with st.spinner("Authentifying..."):
            openai.api_key = os.environ["OPENAI_API_KEY"]
            openai.Model.list()
            deeplake.exists(
                f"hub://{os.environ['ACTIVELOOP_ORG_NAME']}/DataChad-Authentication-Check",
            )
    except Exception as e:
        logger.error(f"Authentication failed with {e}")
        st.session_state["auth_ok"] = False
        st.error("Authentication failed", icon=PAGE_ICON)
        st.stop()

    logger.info("Authentification successful!")
    st.session_state["auth_ok"] = True


def save_uploaded_file(uploaded_file):
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
    logger.info(f"saved {file_path}")
    return file_path


def delete_uploaded_file(uploaded_file):
    # cleanup locally stored files
    file_path = DATA_PATH / uploaded_file.name
    if os.path.exists(DATA_PATH):
        os.remove(file_path)
        logger.info(f"removed {file_path}")


def load_git(data_source):
    # Thank you github for the "master" to "main" switch
    repo_name = data_source.split("/")[-1].split(".")[0]
    repo_path = str(DATA_PATH / repo_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    branches = ["main", "master"]
    for branch in branches:
        if os.path.exists(repo_path):
            data_source = None
        try:
            docs = GitLoader(repo_path, data_source, branch).load_and_split(
                text_splitter
            )
            break
        except Exception as e:
            logger.error(f"error loading git: {e}")
        if os.path.exists(repo_path):
            # cleanup repo afterwards
            shutil.rmtree(repo_path)
    return docs


def load_any_data_source(data_source):
    # ugly thing that decides how to load data
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
        loader = DirectoryLoader(data_source, recursive=True)
    if is_git:
        return load_git(data_source)
    if is_web:
        if is_pdf:
            loader = OnlinePDFLoader(data_source)
        else:
            loader = WebBaseLoader(data_source)
    if is_file:
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
    if loader:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = loader.load_and_split(text_splitter)
        logger.info(f"loaded {len(docs)} document chucks")
        return docs

    error_msg = f"Failed to load {data_source}"
    st.error(error_msg, icon=PAGE_ICON)
    logger.info(error_msg)
    st.stop()


def clean_data_source_string(data_source):
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    dashed_string = re.sub(r"\W+", "-", data_source)
    cleaned_string = re.sub(r"--+", "- ", dashed_string).strip("-")
    return cleaned_string


def setup_vector_store(data_source):
    # either load existing vector store or upload a new one to the hub
    embeddings = OpenAIEmbeddings(disallowed_special=())
    data_source_name = clean_data_source_string(data_source)
    dataset_path = f"hub://{os.environ['ACTIVELOOP_ORG_NAME']}/{data_source_name}"
    if deeplake.exists(dataset_path):
        with st.spinner("Loading vector store..."):
            logger.info(f"{dataset_path} exists -> loading")
            vector_store = DeepLake(
                dataset_path=dataset_path, read_only=True, embedding_function=embeddings
            )
    else:
        with st.spinner("Reading, embedding and uploading data to hub..."):
            logger.info(f"{dataset_path} does not exist -> uploading")
            docs = load_any_data_source(data_source)
            vector_store = DeepLake.from_documents(
                docs,
                embeddings,
                dataset_path=f"hub://{os.environ['ACTIVELOOP_ORG_NAME']}/{data_source_name}",
            )
    return vector_store


def get_chain(data_source):
    # create the langchain that will be called to generate responses
    vector_store = setup_vector_store(data_source)
    retriever = vector_store.as_retriever()
    search_kwargs = {
        "distance_metric": "cos",
        "fetch_k": 20,
        "maximal_marginal_relevance": True,
        "k": 10,
    }
    retriever.search_kwargs.update(search_kwargs)
    model = ChatOpenAI(model_name=MODEL)
    with st.spinner("Building langchain..."):
        chain = ConversationalRetrievalChain.from_llm(
            model,
            retriever=retriever,
            chain_type="stuff",
            verbose=True,
            max_tokens_limit=3375,
        )
        logger.info(f"{data_source} is ready to go!")
    return chain


def build_chain_and_clear_history(data_source):
    # Get chain and store it in the session state
    # Also delete chat history to not confuse the bot with old context
    st.session_state["chain"] = get_chain(data_source)
    st.session_state["chat_history"] = []


def generate_response(prompt):
    # call the chain to generate responses and add them to the chat history
    with st.spinner("Generating response"):
        response = st.session_state["chain"](
            {"question": prompt, "chat_history": st.session_state["chat_history"]}
        )
        logger.info(f"{response=}")
        st.session_state["chat_history"].append((prompt, response["answer"]))
    return response["answer"]
