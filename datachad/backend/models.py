from dataclasses import dataclass
from typing import Any

import streamlit as st
import tiktoken
import litellm
from litellm import completion
from langchain.base_language import BaseLanguageModel
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import Embeddings, OpenAIEmbeddings
from transformers import AutoTokenizer

from datachad.backend.constants import LOCAL_EMBEDDINGS, MODEL_PATH
from datachad.backend.logging import logger


class Enum:
    @classmethod
    def all(cls) -> list[Any]:
        return [v for k, v in cls.__dict__.items() if not k.startswith("_")]


@dataclass
class Model:
    name: str
    embedding: str
    context: int

    def __str__(self) -> str:
        return self.name


class STORES(Enum):
    KNOWLEDGE_BASE = "Knowledge Base"
    SMART_FAQ = "Smart FAQ"


class EMBEDDINGS(Enum):
    # Add more embeddings as needed
    OPENAI = "text-embedding-3-small"
    HUGGINGFACE = "sentence-transformers/all-MiniLM-L6-v2"


class MODELS(Enum):
    # Add more models as needed
    GPT35TURBO = Model(
        name="gpt-3.5-turbo",
        embedding=EMBEDDINGS.OPENAI,
        context=4096,
    )
    GPT35TURBO16K = Model(
        name="gpt-3.5-turbo-16k",
        embedding=EMBEDDINGS.OPENAI,
        context=16385,
    )
    GPT4 = Model(
        name="gpt-4",
        embedding=EMBEDDINGS.OPENAI,
        context=8192,
    )
    GPT4TURBO = Model(
        name="gpt-4-turbo-preview",
        embedding=EMBEDDINGS.OPENAI,
        context=128000,
    )


def get_model(options: dict, credentials: dict) -> BaseLanguageModel:
    match options["model"].name:
        case model_name if model_name.startswith("gpt"):
            model = ChatOpenAI(
                model_name=options["model"].name,
                temperature=options["temperature"],
                openai_api_key=credentials["openai_api_key"],
                streaming=True,
            )
        case *litellm.model_list:
            model = ChatOpenAI(
                model_name=options["model"].name,
                temperature=options["temperature"],
                openai_api_key=credentials["openai_api_key"],
                client=completion
            )
        # Added models need to be cased here
        case _default:
            msg = f"Model {options['model'].name} not supported!"
            logger.error(msg)
            st.error(msg)
            exit
    return model


def get_embeddings(options: dict, credentials: dict) -> Embeddings:
    match options["model"].embedding:
        case embedding if (embedding == EMBEDDINGS.HUGGINGFACE or LOCAL_EMBEDDINGS):
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS.HUGGINGFACE, cache_folder=str(MODEL_PATH)
            )
        case EMBEDDINGS.OPENAI:
            embeddings = OpenAIEmbeddings(
                model=EMBEDDINGS.OPENAI,
                disallowed_special=(),
                openai_api_key=credentials["openai_api_key"],
            )
        # Added embeddings need to be cased here
        case _default:
            msg = f"Embeddings {options['model'].embedding} not supported!"
            logger.error(msg)
            st.error(msg)
            exit
    return embeddings


def get_tokenizer(options: dict) -> Embeddings:
    match options["model"].embedding:
        case embedding if (embedding == EMBEDDINGS.HUGGINGFACE or LOCAL_EMBEDDINGS):
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDINGS.HUGGINGFACE)
        case EMBEDDINGS.OPENAI:
            tokenizer = tiktoken.encoding_for_model(EMBEDDINGS.OPENAI)
        # Added tokenizers need to be cased here
        case _default:
            msg = f"Tokenizer {options['model'].embedding} not supported!"
            logger.error(msg)
            st.error(msg)
            exit
    return tokenizer
