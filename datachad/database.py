import os
import re

import deeplake
import streamlit as st
from langchain.vectorstores import DeepLake, VectorStore

from datachad.constants import DATA_PATH
from datachad.loader import load_data_source
from datachad.models import MODES, get_embeddings
from datachad.utils import logger


def get_dataset_path() -> str:
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    dataset_name = re.sub(r"\W+", "-", st.session_state["data_source"])
    dataset_name = re.sub(r"--+", "- ", dataset_name).strip("-")
    if st.session_state["mode"] == MODES.LOCAL:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        dataset_path = str(DATA_PATH / dataset_name)
    else:
        dataset_path = f"hub://{st.session_state['activeloop_org_name']}/{dataset_name}-{st.session_state['chunk_size']}"
    return dataset_path


def get_vector_store() -> VectorStore:
    # either load existing vector store or upload a new one to the hub
    embeddings = get_embeddings()
    dataset_path = get_dataset_path()
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
