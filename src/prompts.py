

QA_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""


FLASH_CARD_PROMPT_TEMPLATE = """
You are an assistant for text summary tasks.
Use the following context, extract {num_cards} most important information and make a question-answer pair for each. \
DON'T MAKE UP ANYTHING. 
Use exact words from context.

{context}

---
Answer start here
"""
