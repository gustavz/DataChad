from typing import Any

from langchain.callbacks.manager import CallbackManagerForChainRun, Callbacks
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory, BasePromptTemplate, BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from datachad.backend.constants import VERBOSE

from datachad.backend.deeplake import get_or_create_deeplake_vector_store_display_name
from datachad.backend.logging import logger
from datachad.backend.models import get_model
from datachad.backend.prompts import (
    CONDENSE_QUESTION_PROMPT,
    KNOWLEDGE_BASE_PROMPT,
    QA_PROMPT,
    SMART_FAQ_PROMPT,
)


class MultiRetrieverFAQChain(Chain):
    """
    This chain does blablabla
    """

    output_key: str = "answer"
    rephrase_question: bool = True
    use_vanilla_llm: bool = True
    max_tokens_limit: int
    qa_chain: LLMChain
    condense_question_chain: LLMChain
    knowledge_base_chain: BaseCombineDocumentsChain
    knowledge_base_retrievers: list[BaseRetriever]
    smart_faq_chain: BaseCombineDocumentsChain
    smart_faq_retriever: BaseRetriever | None

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the prompt expects."""
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> list[str]:
        """Will always return text key."""
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "stuff"

    def _reduce_tokens_below_limit(
        self, docs: list[Document], combine_docs_chain: BaseCombineDocumentsChain
    ) -> list[Document]:
        num_docs = len(docs)

        tokens = [combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content) for doc in docs]
        token_count = sum(tokens[:num_docs])
        while token_count > self.max_tokens_limit:
            num_docs -= 1
            token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self,
        question: str,
        retriever: BaseRetriever,
        combine_docs_chain: BaseCombineDocumentsChain,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs from retriever."""
        docs = retriever.get_relevant_documents(question, callbacks=run_manager.get_child())
        return self._reduce_tokens_below_limit(docs, combine_docs_chain)

    def _add_text_to_answer(
        self, text: str, answer: str, run_manager: CallbackManagerForChainRun
    ) -> str:
        """Hack to add text to the streaming response handler"""
        answer += text
        streamhandler = next(
            (h for h in run_manager.get_child().handlers if hasattr(h, "stream_text")),
            None,
        )
        if streamhandler:
            streamhandler.on_llm_new_token(text)
        return answer

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        answer = ""
        chat_history_str = _get_chat_history(inputs["chat_history"])
        run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # Generate new standalone question if there is a chat history
        if chat_history_str and self.rephrase_question:
            inputs["question"] = self.condense_question_chain.run(
                question=inputs["question"],
                chat_history=chat_history_str,
                callbacks=run_manager.get_child(),
            )
        # Answer the question using the FAQ document context
        if self.smart_faq_retriever:
            docs = self._get_docs(
                inputs["question"],
                self.smart_faq_retriever,
                self.smart_faq_chain,
                run_manager=run_manager,
            )
            smart_faq_name = get_or_create_deeplake_vector_store_display_name(
                self.smart_faq_retriever.vectorstore.dataset_path
            )
            answer = self._add_text_to_answer(
                f"\n#### SMART FAQ ANSWER `{smart_faq_name}`\n", answer, run_manager
            )
            answer += self.smart_faq_chain.run(
                input_documents=docs, callbacks=run_manager.get_child(), **inputs
            )

        # Answer the question using all provided knowledge bases
        for i, retriever in enumerate(self.knowledge_base_retrievers):
            docs = self._get_docs(
                inputs["question"],
                retriever,
                self.knowledge_base_chain,
                run_manager=run_manager,
            )
            knowledge_base_name = get_or_create_deeplake_vector_store_display_name(
                retriever.vectorstore.dataset_path
            )
            answer = self._add_text_to_answer(
                f"\n#### KNOWLEDGE BASE ANSWER `{knowledge_base_name}`\n",
                answer,
                run_manager,
            )
            answer += self.knowledge_base_chain.run(
                input_documents=docs, callbacks=run_manager.get_child(), **inputs
            )
        # Answer the question using
        # the general purpose QA chain
        if self.use_vanilla_llm:
            answer = self._add_text_to_answer("\n#### LLM ANSWER\n", answer, run_manager)
            answer += self.qa_chain.run(
                question=inputs["question"], callbacks=run_manager.get_child()
            )
        return {self.output_key: answer}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        condense_question_prompt: BasePromptTemplate,
        smart_faq_prompt: BasePromptTemplate,
        knowledge_base_prompt: BasePromptTemplate,
        qa_prompt: BasePromptTemplate,
        knowledge_base_retrievers: list[BaseRetriever],
        smart_faq_retriever: BaseRetriever | None = None,
        retriever_llm: BaseLanguageModel | None = None,
        condense_question_llm: BaseLanguageModel | None = None,
        use_vanilla_llm: bool = True,
        callbacks: Callbacks = None,
        chain_type: str = "stuff",
        verbose: bool = False,
        **kwargs: Any,
    ) -> Chain:
        qa_chain = LLMChain(
            llm=llm,
            prompt=qa_prompt,
            callbacks=callbacks,
            verbose=verbose,
        )
        condense_question_chain = LLMChain(
            llm=condense_question_llm or llm,
            prompt=condense_question_prompt,
            callbacks=callbacks,
            verbose=verbose,
        )
        knowledge_base_chain = load_qa_chain(
            llm=retriever_llm or llm,
            prompt=knowledge_base_prompt,
            chain_type=chain_type,
            callbacks=callbacks,
            verbose=verbose,
        )
        smart_faq_chain = load_qa_chain(
            llm=retriever_llm or llm,
            prompt=smart_faq_prompt,
            chain_type=chain_type,
            callbacks=callbacks,
            verbose=verbose,
        )
        return cls(
            qa_chain=qa_chain,
            condense_question_chain=condense_question_chain,
            knowledge_base_chain=knowledge_base_chain,
            knowledge_base_retrievers=knowledge_base_retrievers,
            smart_faq_chain=smart_faq_chain,
            smart_faq_retriever=smart_faq_retriever,
            use_vanilla_llm=use_vanilla_llm,
            callbacks=callbacks,
            **kwargs,
        )


def get_knowledge_base_search_kwargs(options: dict) -> tuple[dict, str]:
    k = int(options["max_tokens"] // options["chunk_size"])
    fetch_k = k * options["k_fetch_k_ratio"]
    if options["maximal_marginal_relevance"]:
        search_kwargs = {
            "distance_metric": options["distance_metric"],
            "fetch_k": fetch_k,
            "k": k,
        }
        search_type = "mmr"
    else:
        search_kwargs = {
            "k": k,
            "distance_metric": options["distance_metric"],
        }
        search_type = "similarity"

    return search_kwargs, search_type


def get_smart_faq_search_kwargs(options: dict) -> tuple[dict, str]:
    search_kwargs = {
        "k": 20,
        "distance_metric": options["distance_metric"],
    }
    search_type = "similarity"
    return search_kwargs, search_type


def get_multi_chain(
    use_vanilla_llm: bool,
    knowledge_bases: list[VectorStore],
    smart_faq: VectorStore,
    chat_history: BaseChatMessageHistory,
    options: dict,
    credentials: dict,
) -> MultiRetrieverFAQChain:
    kb_search_kwargs, search_type = get_knowledge_base_search_kwargs(options)
    kb_retrievers = [
        kb.as_retriever(search_type=search_type, search_kwargs=kb_search_kwargs)
        for kb in knowledge_bases
    ]
    faq_search_kwargs, search_type = get_smart_faq_search_kwargs(options)
    faq_retriever = (
        smart_faq.as_retriever(search_type=search_type, search_kwargs=faq_search_kwargs)
        if smart_faq
        else None
    )
    model = get_model(options, credentials)
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=chat_history, return_messages=True
    )
    chain = MultiRetrieverFAQChain.from_llm(
        llm=model,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        knowledge_base_prompt=KNOWLEDGE_BASE_PROMPT,
        smart_faq_prompt=SMART_FAQ_PROMPT,
        qa_prompt=QA_PROMPT,
        knowledge_base_retrievers=kb_retrievers,
        smart_faq_retriever=faq_retriever,
        max_tokens_limit=options["max_tokens"],
        use_vanilla_llm=use_vanilla_llm,
        memory=memory,
        verbose=VERBOSE,
    )
    logger.info(f"Multi chain with settings {options} build!")
    return chain
