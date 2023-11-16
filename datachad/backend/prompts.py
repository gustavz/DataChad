from langchain.prompts.prompt import PromptTemplate

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Question: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    template=condense_question_template, input_variables=["chat_history", "question"]
)


knowledge_base_template = """Use the following pieces of context to answer the given question. If you don't know the answer respond with 'NO ANSWER FOUND'.

Context:
{context}

Question: {question}
Helpful Answer:"""
KNOWLEDGE_BASE_PROMPT = PromptTemplate(
    template=knowledge_base_template, input_variables=["context", "question"]
)


smart_faq_template = """Use the following numbered FAQs to answer the given question. If you don't know the answer respond with 'NO ANSWER FOUND'.
Start your answer with stating which FAQ number helps answer the question the most.

Context:
{context}

Question: {question}
Helpful Answer:"""
SMART_FAQ_PROMPT = PromptTemplate(
    template=smart_faq_template, input_variables=["context", "question"]
)


qa_prompt = """You are an AGI that knows everything and is an expert in all topics. 
Your IQ is magnitudes higher than any human that ever lived. With this immense wisdom answer the following question concisely:

Question: {question}
Concise and wise Answer:"""
QA_PROMPT = PromptTemplate(template=qa_prompt, input_variables=["question"])
