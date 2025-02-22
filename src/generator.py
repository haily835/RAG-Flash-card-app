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

from typing import List
import pandas as pd
# Load environment variables (API keys)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for text summary tasks.
Use the following context, extract {num_cards} most important information and make a quesion-answer pair for each. \
DON'T MAKE UP ANYTHING. 
Use exact words from context.

{context}

---
Answer start here
"""


class KeypointWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
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


def generate_response(context, num_cards):

    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """
    llm = ChatOpenAI(model="gpt-4o-mini")


    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.invoke({
        "context": format_docs(context),
        "num_cards": num_cards
    })
    gpt = llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    structured_response = gpt.invoke(prompt)
   
    df = pd.DataFrame([structured_response.dict()])
    return df

    # # Transforming into a table with two rows: 'answer' and 'source'
    # answer_row = []
    # source_row = []
    # reasoning_row = []

    # for col in df.columns:
    #     answer_row.append(df[col][0]['answer'])
    #     source_row.append(df[col][0]['sources'])
    #     reasoning_row.append(df[col][0]['reasoning'])

    # # Create new dataframe with two rows: 'answer' and 'source'
    # structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])
  
    # return structured_response_df.T




if __name__ == "__main__":
    user_query = input("Enter your interview question: ")

    # Retrieve relevant documents
    retrieved_docs = retrieve_context(user_query, k=3)

    if not retrieved_docs:
        print("\nNo relevant documents found. Answering without context...")
        retrieved_docs = ["No additional context available."]

    # Generate LLM response
    answer = generate_response(user_query, retrieved_docs)

    # Print results
    print("\n🔹 **AI Response:**")
    print(answer)

    print("\n📚 **Sources:**")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc[:300]}...")  # Show first 300 chars of each source
