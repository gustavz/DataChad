from datetime import datetime
from glob import glob

import deeplake
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.bugout_reporter import deeplake_reporter
from langchain.schema import Document
from langchain.vectorstores import DeepLake, VectorStore

from datachad.backend.constants import DATA_PATH, DEFAULT_USER, LOCAL_DEEPLAKE, STORE_DOCS_EXTRA
from datachad.backend.io import clean_string_for_storing
from datachad.backend.loader import load_data_source, split_docs
from datachad.backend.logging import logger
from datachad.backend.models import STORES, get_embeddings
from datachad.backend.utils import clean_string_for_storing

SPLIT = "-_-"


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
            res_datasets = self.get_workspace_datasets(workspace, suffix_public, suffix_user)
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


def get_deeplake_dataset_path(dataset_name: str, credentials: dict) -> str:
    if LOCAL_DEEPLAKE:
        dataset_path = str(DATA_PATH / dataset_name)
    else:
        dataset_path = f"hub://{credentials['activeloop_id']}/{dataset_name}"
    return dataset_path


def delete_all_deeplake_datasets(credentials: dict) -> None:
    datasets = list_deeplake_datasets(credentials["activeloop_id"], credentials["activeloop_token"])
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


def get_or_create_deeplake_vector_store_paths_for_user(
    credentials: dict, store_type: str
) -> list[str]:
    all_paths = get_existing_deeplake_vector_store_paths(credentials)
    # TODO: replace DEFAULT_USER with user id once stored in credentials
    user_paths = [
        p
        for p in all_paths
        if p.split(SPLIT)[-1] == DEFAULT_USER and p.split(SPLIT)[-2] == store_type
    ]
    return user_paths


def get_or_create_deeplake_vector_store_display_name(dataset_path: str) -> str:
    splits = dataset_path.split(SPLIT)
    return f"{splits[-4]} ({splits[-3][:4]}-{splits[-3][4:6]}-{splits[-3][6:8]})"


def get_unique_deeplake_vector_store_path(store_type: str, name: str, credentials: dict) -> str:
    store_type_dict = {STORES.KNOWLEDGE_BASE: "kb", STORES.SMART_FAQ: "faq"}
    dataset_name = (
        # [-4] vector store name
        f"{SPLIT}{name}"
        # [-3]: creation time
        f"{SPLIT}{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # [-2]: vector store type
        f"{SPLIT}{store_type_dict[store_type]}"
        # [-1]: user
        f"{SPLIT}{DEFAULT_USER}"
    )
    dataset_path = get_deeplake_dataset_path(dataset_name, credentials)
    return dataset_path


def get_deeplake_docs_path(data_source: str, options: dict, credentials: dict) -> str:
    dataset_name = clean_string_for_storing(data_source)
    dataset_name += "-docs"
    dataset_path = get_deeplake_dataset_path(dataset_name, options, credentials)
    return dataset_path


def load_docs_from_deeplake(docs_path: str, credentials: dict) -> list[Document]:
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


def store_docs_to_deeplake(docs: list[Document], docs_path: str, credentials: dict):
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


def load_data_sources_or_docs_from_deeplake(
    data_sources: list[str], options: dict, credentials: dict
) -> list[Document]:
    docs = []
    for data_source in data_sources:
        if STORE_DOCS_EXTRA:
            docs_path = get_deeplake_docs_path(data_source, options, credentials)
            if deeplake.exists(docs_path, token=credentials["activeloop_token"]):
                logger.info(f"Docs exist -> loading docs: {docs_path}")
                docs.extend(load_docs_from_deeplake(docs_path, credentials))
            else:
                logger.info(
                    f"Docs do not exist for data source -> loading data source: {data_source}"
                )
                docs.extend(load_data_source(data_source))
                store_docs_to_deeplake(docs, docs_path, credentials)
            logger.info(f"Docs {docs_path} loaded!")
        else:
            docs.extend(load_data_source(data_source))
    return docs


def get_or_create_deeplake_vector_store(
    data_sources: list[str],
    vector_store_path: str,
    store_type: str,
    options: dict,
    credentials: dict,
) -> VectorStore:
    embeddings = get_embeddings(options, credentials)
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
        docs = load_data_sources_or_docs_from_deeplake(data_sources, options, credentials)
        docs = split_docs(docs, store_type, options)
        vector_store = DeepLake.from_documents(
            docs,
            embeddings,
            dataset_path=vector_store_path,
            token=credentials["activeloop_token"],
        )
    logger.info(f"Vector Store {vector_store_path} loaded!")
    return vector_store
