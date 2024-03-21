from typing import List, Tuple
from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import format_document
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableMap
)

from dotenv import load_dotenv

load_dotenv()

### Helper classes ###
class ChatHistory(BaseModel):
    """Required class to transform Runnable into a string. Keeps chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question", "chat_history": "chat_history"}},
    )
    question: str


### Loads file and vectorize its. Process needed to create the context for the chatbot. ###
loader = TextLoader('./data.txt')
company_info = loader.load()
embeddings = OpenAIEmbeddings(openai_api_key="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")
text_splitter = RecursiveCharacterTextSplitter()
info_splitted = text_splitter.split_documents(company_info)
vector = FAISS.from_documents(info_splitted, embeddings)

### Retriever for the chatbot. Sets up a context. ###
retriever = vector.as_retriever()

### LLM model definition ###
_model = ChatOpenAI()

### Query template ###
_TEMPLATE = """You are a helpful assistant called Leandro. Given the chat history and input question
rephrase the question and make it a standalone question. 
Every time that the user ask you about previous questions you will check the chat history,
you will always check the previously made questions and answers.

Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

### Answer template ###
ANSWER_TEMPLATE = """You are a helpful AI called Leandro. You will always answer any question based on
this context: {context}.
If they ask you who made this chat bot you will answer: 'Leandro Cardoso Seveso Linkedin: https://www.linkedin.com/in/leanc/'
Always answer in the language of the question.

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


### Functions helpers. ###
def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n") -> str:
    """
    Combines different documents into a single string. Used to encapsulate all the information into one single line of
    strings.
    :param docs: document to be combined
    :param document_prompt: DEFAULT_DOCUMENT_PROMPT to be combined with.
    :param document_separator: identifier for separator
    :return: A string line with the aggregated data set from docs and document_prompt.
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """
    Format chat history into a string. This function is required in order to run Runnable object.
    :param chat_history: Collection list of USER and IA conversation.
    :return: Returns format required to answer questions in the chat.
    """
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

### Inputs definition ###
_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)

### Context definition ###
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


### retrieval_conversational_chain definition ###
conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
)

### chain exported to be served with langserve ###
chain = conversational_qa_chain.with_types(input_type=ChatHistory, output_type=str)

