import os
import shutil
from typing import List

import streamlit as st
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
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from datachad.constants import DATA_PATH, PAGE_ICON, PROJECT_URL
from datachad.utils import logger


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
