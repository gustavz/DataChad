import io

from langchain.chains.base import Chain
from langchain.schema import BaseChatMessageHistory
from langchain.schema.vectorstore import VectorStore

from datachad.backend.chain import get_multi_chain
from datachad.backend.deeplake import (
    get_or_create_deeplake_vector_store,
    get_unique_deeplake_vector_store_path,
)
from datachad.backend.io import delete_files, save_files
from datachad.backend.models import STORES


def create_vector_store(
    files: list[io.BytesIO],
    store_type: str,
    name: str,
    options: dict,
    credentials: dict,
) -> VectorStore:
    data_source = save_files(files)
    vector_store_path = get_unique_deeplake_vector_store_path(store_type, name, credentials)
    vector_store = get_or_create_deeplake_vector_store(
        data_source=data_source,
        vector_store_path=vector_store_path,
        store_type=store_type,
        options=options,
        credentials=credentials,
    )
    delete_files(files)
    return vector_store


def create_chain(
    use_vanilla_llm: bool,
    knowledge_bases: str,
    smart_faq: str,
    chat_history: BaseChatMessageHistory,
    options: dict,
    credentials: dict,
) -> Chain:
    knowledge_bases = [
        get_or_create_deeplake_vector_store(
            data_source=None,
            vector_store_path=path,
            store_type=STORES.KNOWLEDGE_BASE,
            options=options,
            credentials=credentials,
        )
        for path in knowledge_bases
    ]
    if smart_faq:
        smart_faq = get_or_create_deeplake_vector_store(
            data_source=None,
            vector_store_path=smart_faq,
            store_type=STORES.SMART_FAQ,
            options=options,
            credentials=credentials,
        )
    chain = get_multi_chain(
        use_vanilla_llm, knowledge_bases, smart_faq, chat_history, options, credentials
    )
    return chain
