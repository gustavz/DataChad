from dataclasses import dataclass
from typing import Any, List

import streamlit as st
import tiktoken
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import Embeddings, OpenAIEmbeddings
from transformers import AutoTokenizer

from datachad.backend.constants import MODEL_PATH
from datachad.backend.logging import logger


class Enum:
    @classmethod
    def all(cls) -> List[Any]:
        return [v for k, v in cls.__dict__.items() if not k.startswith("_")]


@dataclass
class Model:
    name: str
    embedding: str
    path: str = None  # for local models only

    def __str__(self) -> str:
        return self.name


class EMBEDDINGS(Enum):
    # Add more embeddings as needed
    OPENAI = "text-embedding-ada-002"
    HUGGINGFACE = "sentence-transformers/all-MiniLM-L6-v2"


class MODELS(Enum):
    # Add more models as needed
    GPT35TURBO = Model(
        name="gpt-3.5-turbo",
        embedding=EMBEDDINGS.OPENAI,
    )
    GPT4 = Model(name="gpt-4", embedding=EMBEDDINGS.OPENAI)


def get_model(options: dict, credentials: dict) -> BaseLanguageModel:
    match options["model"].name:
        case MODELS.GPT35TURBO.name | MODELS.GPT4.name:
            model = ChatOpenAI(
                model_name=options["model"].name,
                temperature=options["temperature"],
                openai_api_key=credentials["openai_api_key"],
                streaming=True,
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
        case EMBEDDINGS.OPENAI:
            embeddings = OpenAIEmbeddings(
                model=EMBEDDINGS.OPENAI,
                disallowed_special=(),
                openai_api_key=credentials["openai_api_key"],
            )
        case EMBEDDINGS.HUGGINGFACE:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS.HUGGINGFACE, cache_folder=str(MODEL_PATH)
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
        case EMBEDDINGS.OPENAI:
            tokenizer = tiktoken.encoding_for_model(EMBEDDINGS.OPENAI)
        case EMBEDDINGS.HUGGINGFACE:
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDINGS.HUGGINGFACE)
        # Added tokenizers need to be cased here
        case _default:
            msg = f"Tokenizer {options['model'].embedding} not supported!"
            logger.error(msg)
            st.error(msg)
            exit
    return tokenizer
