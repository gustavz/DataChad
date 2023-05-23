import streamlit as st
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.chains import ConversationalRetrievalChain

from datachad.constants import PAGE_ICON
from datachad.database import get_vector_store
from datachad.models import get_model
from datachad.utils import logger


def get_chain() -> ConversationalRetrievalChain:
    # create the langchain that will be called to generate responses
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()
    # Search params "fetch_k" and "k" define how many documents are pulled from the hub
    # and selected after the document matching to build the context
    # that is fed to the model together with your prompt
    search_kwargs = {
        "maximal_marginal_relevance": True,
        "distance_metric": "cos",
        "fetch_k": st.session_state["fetch_k"],
        "k": st.session_state["k"],
    }
    retriever.search_kwargs.update(search_kwargs)
    model = get_model()
    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        # we limit the maximum number of used tokens
        # to prevent running into the models token limit of 4096
        max_tokens_limit=st.session_state["max_tokens"],
    )
    return chain


def update_chain() -> None:
    # Build chain with parameters from session state and store it back
    # Also delete chat history to not confuse the bot with old context
    try:
        st.session_state["chain"] = get_chain()
        st.session_state["chat_history"] = []
        msg = f"Data source '{st.session_state['data_source']}' is ready to go!"
        logger.info(msg)
        st.info(msg, icon=PAGE_ICON)
    except Exception as e:
        msg = f"Failed to build chain for data source '{st.session_state['data_source']}' with error: {e}"
        logger.error(msg)
        st.error(msg, icon=PAGE_ICON)


def update_usage(cb: OpenAICallbackHandler) -> None:
    # Accumulate API call usage via callbacks
    logger.info(f"Usage: {cb}")
    callback_properties = [
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_cost",
    ]
    for prop in callback_properties:
        value = getattr(cb, prop, 0)
        st.session_state["usage"].setdefault(prop, 0)
        st.session_state["usage"][prop] += value


def generate_response(prompt: str) -> str:
    # call the chain to generate responses and add them to the chat history
    with st.spinner("Generating response"), get_openai_callback() as cb:
        response = st.session_state["chain"](
            {"question": prompt, "chat_history": st.session_state["chat_history"]}
        )
        update_usage(cb)
    logger.info(f"Response: '{response}'")
    st.session_state["chat_history"].append((prompt, response["answer"]))
    return response["answer"]
