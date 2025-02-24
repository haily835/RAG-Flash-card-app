import os
import chromadb
import openai
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from prompts import FLASH_CARD_PROMPT_TEMPLATE, QA_PROMPT_TEMPLATE
from typing import List
import pandas as pd

class KeypointWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    question: str = Field(description="Question for the keypoint")
    keypoint: str = Field(description="Keypoint from text")
    sources: str = Field(description="Full direct text chunk from the context for this keypoint")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    

class ExtractedInfoWithSources(BaseModel):
    """Extracted information about the research article"""
    cards: List[KeypointWithSources]

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)


def generate_flashcard(context, num_cards, api_key):
    """Create flash card from given context"""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


    prompt_template = ChatPromptTemplate.from_template(FLASH_CARD_PROMPT_TEMPLATE)
    prompt = prompt_template.invoke({
        "context": format_docs(context),
        "num_cards": num_cards
    })
    gpt = llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    structured_response = gpt.invoke(prompt)
   
    return structured_response


class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    

def generate_question_answering(context, question, api_key):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    prompt_template = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
    prompt = prompt_template.invoke({
        "context": format_docs(context),
        "question": question
    })
    gpt = llm.with_structured_output(AnswerWithSources, strict=True)
    structured_response = gpt.invoke(prompt)


    return structured_response

