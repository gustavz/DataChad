from glob import glob
from typing import List

import deeplake
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.bugout_reporter import deeplake_reporter
from langchain.schema import Document
from langchain.vectorstores import DeepLake, VectorStore

from datachad.backend.constants import DATA_PATH, DEFAULT_USER, LOCAL_DEEPLAKE
from datachad.backend.io import clean_string_for_storing
from datachad.backend.loader import load_data_source, split_docs
from datachad.backend.logging import logger
from datachad.backend.models import get_embeddings
from datachad.backend.utils import clean_string_for_storing

SPLIT = "_"


def list_deeplake_datasets(
    org_id: str = "",
    token: str = None,
) -> None:
    """List all available Deep Lake cloud datasets for a given user / orgnaization.
    Removed from deeplake in: https://github.com/activeloopai/deeplake/pull/2182/files
    """

    deeplake_reporter.feature_report(
        feature_name="list",
        parameters={"org_id": org_id},
    )

    def get_datasets(self, workspace: str):
        LIST_DATASETS = "/api/datasets/{}"
        suffix_public = LIST_DATASETS.format("public")
        suffix_user = LIST_DATASETS.format("all")
        if workspace:
            res_datasets = self.get_workspace_datasets(
                workspace, suffix_public, suffix_user
            )
        else:
            public_datasets = self.request(
                "GET",
                suffix_public,
                endpoint=self.endpoint(),
            ).json()
            user_datasets = self.request(
                "GET",
                suffix_user,
                endpoint=self.endpoint(),
            ).json()
            res_datasets = public_datasets + user_datasets
        return [ds["_id"] for ds in res_datasets]

    client = DeepLakeBackendClient(token=token)
    client.get_datasets = get_datasets
    datasets = client.get_datasets(client, workspace=org_id)
    return datasets


def get_deeplake_dataset_path(dataset_name: str, options: dict, credentials: dict):
    # TODO add user id and dataset size as unique id
    if LOCAL_DEEPLAKE:
        dataset_path = str(DATA_PATH / dataset_name)
    else:
        dataset_path = f"hub://{credentials['activeloop_id']}/{dataset_name}"
    return dataset_path


def delete_all_deeplake_datasets(credentials: dict):
    datasets = list_deeplake_datasets(
        credentials["activeloop_id"], credentials["activeloop_token"]
    )
    for dataset in datasets:
        path = f"hub://{dataset}"
        logger.info(f"Deleting dataset: {path}")
        deeplake.delete(path, token=credentials["activeloop_token"], force=True)


def get_existing_deeplake_vector_store_paths(credentials: dict) -> list[str]:
    if LOCAL_DEEPLAKE:
        return glob(str(DATA_PATH / "*"), recursive=False)
    else:
        dataset_names = list_deeplake_datasets(
            credentials["activeloop_id"], credentials["activeloop_token"]
        )
        dataset_pahs = [f"hub://{name}" for name in dataset_names]
        return dataset_pahs


def get_deeplake_vector_store_paths_for_user(credentials: dict) -> list[str]:
    all_paths = get_existing_deeplake_vector_store_paths(credentials)
    # TODO: replace DEFAULT_USER with user id once supported
    user_paths = [p for p in all_paths if p.split(SPLIT)[-1] == DEFAULT_USER]
    return user_paths


def get_data_source_from_deeplake_dataset_path(dataset_path):
    data_source = (
        f"{SPLIT}".join(dataset_path.split(SPLIT)[:-3]).split("/")[-1].lstrip("data-")
    )
    return data_source


def get_deeplake_vector_store_path(
    data_source: str, options: dict, credentials: dict
) -> str:
    dataset_name = (
        f"{clean_string_for_storing(data_source)}"
        f"{SPLIT}{options['chunk_size']}-{options['chunk_overlap_pct']}"
        f"{SPLIT}{options['model'].embedding}"
        # TODO: replace DEFAULT_USER with user id once supported
        f"{SPLIT}{DEFAULT_USER}"
    )
    dataset_path = get_deeplake_dataset_path(dataset_name, options, credentials)
    return dataset_path


def get_deeplake_docs_path(data_source: str, options: dict, credentials: dict) -> str:
    dataset_name = clean_string_for_storing(data_source)
    dataset_name += "-docs"
    dataset_path = get_deeplake_dataset_path(dataset_name, options, credentials)
    return dataset_path


def load_docs_from_deeplake(docs_path: str, credentials: dict) -> List[Document]:
    ds = deeplake.load(docs_path, token=credentials["activeloop_token"])
    metadatas = ds["metadata"].data()["value"]
    texts = ds["text"].data()["value"]
    docs = [
        Document(
            page_content=text,
            metadata=metadata,
        )
        for text, metadata in zip(texts, metadatas)
    ]
    return docs


def store_docs_to_deeplake(docs: List[Document], docs_path: str, credentials: dict):
    ds = deeplake.empty(docs_path, token=credentials["activeloop_token"])
    ds.create_tensor(
        "text",
        htype="text",
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        chunk_compression="lz4",
    )
    ds.create_tensor(
        "metadata",
        htype="json",
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        chunk_compression="lz4",
    )
    for doc in docs:
        ds.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
        )
    ds.commit()
    logger.info(f"Stored docs to: {docs_path}")


def load_data_source_or_docs_from_deeplake(
    data_source: str, options: dict, credentials: dict
) -> List[Document]:
    if options["store_docs_extra"]:
        docs_path = get_deeplake_docs_path(data_source, options, credentials)
        if deeplake.exists(docs_path, token=credentials["activeloop_token"]):
            logger.info(f"Docs exist -> loading docs: {docs_path}")
            docs = load_docs_from_deeplake(docs_path, credentials)
        else:
            logger.info(
                f"Docs do not exist for data source -> loading data source: {data_source}"
            )
            docs = load_data_source(data_source)
            store_docs_to_deeplake(docs, docs_path, credentials)
        logger.info(f"Docs {docs_path} loaded!")
    else:
        docs = load_data_source(data_source)
    return docs


def get_deeplake_vector_store(
    data_source: str, vector_store_path: str, options: dict, credentials: dict
) -> VectorStore:
    # either load existing vector store or upload a new one to the hub
    embeddings = get_embeddings(options, credentials)
    if not vector_store_path:
        vector_store_path = get_deeplake_vector_store_path(
            data_source, options, credentials
        )
    if deeplake.exists(vector_store_path, token=credentials["activeloop_token"]):
        logger.info(f"Vector Store '{vector_store_path}' exists -> loading")
        vector_store = DeepLake(
            dataset_path=vector_store_path,
            read_only=True,
            embedding_function=embeddings,
            token=credentials["activeloop_token"],
        )
    else:
        logger.info(f"Vector Store '{vector_store_path}' does not exist -> uploading")
        docs = load_data_source_or_docs_from_deeplake(data_source, options, credentials)
        docs = split_docs(docs, options)
        vector_store = DeepLake.from_documents(
            docs,
            embeddings,
            dataset_path=vector_store_path,
            token=credentials["activeloop_token"],
        )
    logger.info(f"Vector Store {vector_store_path} loaded!")
    return vector_store
