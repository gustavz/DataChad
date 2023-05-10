import os
import re

import deeplake
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

from constants import DATA_PATH, MODEL, PAGE_ICON


def validate_keys(openai_key, activeloop_token, activeloop_org_name):
    # Validate all API related variables are set and correct
    # TODO: Do proper token/key validation, currently activeloop has none
    all_keys = [openai_key, activeloop_token, activeloop_org_name]
    if any(all_keys):
        print(f"{openai_key=}\n{activeloop_token=}\n{activeloop_org_name=}")
        if not all(all_keys):
            st.session_state["auth_ok"] = False
            st.error("Authentication failed", icon=PAGE_ICON)
            st.stop()
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
        os.environ["ACTIVELOOP_ORG_NAME"] = activeloop_org_name
    else:
        # Fallback for local development or deployments with provided credentials
        # either env variables or streamlit secrets need to be set
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
            os.environ["ACTIVELOOP_ORG_NAME"] = st.secrets.get("ACTIVELOOP_ORG_NAME")
    st.session_state["auth_ok"] = True


def save_uploaded_file(uploaded_file):
    # streamlit uploaded files need to be stored locally before
    # TODO: delete local files after they are uploaded to the datalake
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    file_path = str(DATA_PATH / uploaded_file.name)
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    file = open(file_path, "wb")
    file.write(file_bytes)
    file.close()
    return file_path


def load_git(data_source):
    # Thank you github for the "master" to "main" switch
    repo_name = data_source.split("/")[-1].split(".")[0]
    repo_path = str(DATA_PATH / repo_name)
    if os.path.exists(repo_path):
        data_source = None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    branches = ["main", "master"]
    for branch in branches:
        try:
            docs = GitLoader(repo_path, data_source, branch).load_and_split(
                text_splitter
            )
        except Exception as e:
            print(f"error loading git: {e}")
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
        print(f"loaded {len(docs)} document chucks")
        return docs

    error_msg = f"Failed to load {data_source}"
    st.error(error_msg, icon=PAGE_ICON)
    print(error_msg)
    st.stop()


def clean_data_source_string(data_source):
    # replace all non-word characters with dashes
    # to get a string that can be used to create a datalake dataset
    dashed_string = re.sub(r"\W+", "-", data_source)
    cleaned_string = re.sub(r"--+", "- ", dashed_string).strip("-")
    return cleaned_string


def setup_vector_store(data_source):
    # either load existing vector store or upload a new one to the datalake
    embeddings = OpenAIEmbeddings(disallowed_special=())
    data_source_name = clean_data_source_string(data_source)
    dataset_path = f"hub://{os.environ['ACTIVELOOP_ORG_NAME']}/{data_source_name}"
    if deeplake.exists(dataset_path):
        print(f"{dataset_path} exists -> loading")
        vector_store = DeepLake(
            dataset_path=dataset_path, read_only=True, embedding_function=embeddings
        )
    else:
        print(f"{dataset_path} does not exist -> uploading")
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
    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        max_tokens_limit=3375,
    )
    print(f"{data_source} is ready to go!")
    return chain


def reset_data_source(data_source):
    # we need to reset all caches if a new data source is loaded
    # otherwise the langchain is confused and produces garbage
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["chat_history"] = []
    st.session_state["chain"] = get_chain(data_source)


def generate_response(prompt):
    # call the chain to generate responses and add them to the chat history
    response = st.session_state["chain"](
        {"question": prompt, "chat_history": st.session_state["chat_history"]}
    )
    print(f"{response=}")
    st.session_state["chat_history"].append((prompt, response["answer"]))
    return response["answer"]
