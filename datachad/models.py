from dataclasses import dataclass

import streamlit as st
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import Embeddings, OpenAIEmbeddings
from langchain.llms import GPT4All, LlamaCpp

from datachad.utils import logger


class Enum:
    @classmethod
    def values(cls):
        return [v for k, v in cls.__dict__.items() if not k.startswith("_")]

    @classmethod
    def dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("_")}


@dataclass
class Model:
    name: str
    mode: str
    embedding: str
    path: str = None  # for local models only

    def __str__(self):
        return self.name


class MODES(Enum):
    OPENAI = "OpenAI"
    LOCAL = "Local"


class EMBEDDINGS(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "all-MiniLM-L6-v2"


class MODELS(Enum):
    GPT35TURBO = Model("gpt-3.5-turbo", MODES.OPENAI, EMBEDDINGS.OPENAI)
    GPT4 = Model("gpt-4", MODES.OPENAI, EMBEDDINGS.OPENAI)
    LLAMACPP = Model(
        "LLAMA", MODES.LOCAL, EMBEDDINGS.HUGGINGFACE, "models/llamacpp.bin"
    )
    GPT4ALL = Model(
        "GPT4All", MODES.LOCAL, EMBEDDINGS.HUGGINGFACE, "models/gpt4all.bin"
    )

    @classmethod
    def for_mode(cls, mode):
        return [v for v in cls.values() if isinstance(v, Model) and v.mode == mode]


def get_model() -> BaseLanguageModel:
    match st.session_state["model"].name:
        case MODELS.GPT35TURBO.name:
            model = ChatOpenAI(
                model_name=st.session_state["model"].name,
                temperature=st.session_state["temperature"],
                openai_api_key=st.session_state["openai_api_key"],
            )
        case MODELS.GPT4.name:
            model = ChatOpenAI(
                model_name=st.session_state["model"].name,
                temperature=st.session_state["temperature"],
                openai_api_key=st.session_state["openai_api_key"],
            )
        case MODELS.LLAMACPP.name:
            model = LlamaCpp(
                model_path=st.session_state["model"].path,
                n_ctx=st.session_state["model_n_ctx"],
                temperature=st.session_state["temperature"],
                verbose=True,
            )
        case MODELS.GPT4ALL.name:
            model = GPT4All(
                model=st.session_state["model"].path,
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
    match st.session_state["model"].embedding:
        case EMBEDDINGS.OPENAI:
            embeddings = OpenAIEmbeddings(
                disallowed_special=(), openai_api_key=st.session_state["openai_api_key"]
            )
        case EMBEDDINGS.HUGGINGFACE:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS.HUGGINGFACE)
        # Add more embeddings as needed
        case _default:
            msg = f"Embeddings {st.session_state['embeddings']} not supported!"
            logger.error(msg)
            st.error(msg)
            exit
    return embeddings
