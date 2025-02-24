# RAG QA and Flash Card App

RAG Flash Card App i  Retrieval-Augmented Generation (RAG) project that extracts key points from PDF files and generates flashcards from it using OpenAI api, langchain, chromadb and streamlit.

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a hybrid approach that combines information retrieval with generative language models to produce more accurate, contextually enriched responses. Instead of relying solely on the model’s internal knowledge, RAG first retrieves relevant documents or passages from an external knowledge base (using tools like FAISS or ChromaDB) and then feeds that context into a generative model (like GPT) to produce a well-informed answer.

## Who will need this? 
- 🎓 Students
Use course slides and textbooks as a knowledge base for LLM models, enabling them to generate accurate answers without fabricating information.

- 🔬 Researchers
Summarize and extract key points from recently published research papers, ensuring the essence of cutting-edge findings is captured.


## Tech Stack
🐍 Python
The primary programming language for developing the project.

⚡ Streamlit
A framework for building the web application, providing an intuitive UI for interaction.

📚 Chroma
A database used for storing embeddings, enabling efficient retrieval of context.

🔗 LangChain
Utilized for reading PDFs, extracting text, and converting it into embeddings via the OpenAI API.

🐳 Docker
Containerizes the development environment to ensure consistency and reproducibility.

💻 VS Code Dev Containers
An extension that facilitates development within a containerized environment.



## Project Structure
```bash
.
├── README.md
├── data // Pdf file uploaded.
├── db // Chroma db
├── models // Folder to store embedding.
├── notebooks
│   └── exploration.ipynb
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── generator.cpython-312.pyc
│   │   ├── ingest.cpython-312.pyc
│   │   ├── prompts.cpython-312.pyc
│   │   ├── retriever.cpython-310.pyc
│   │   └── retriever.cpython-312.pyc
│   ├── app.py
│   ├── generator.py
│   ├── ingest.py
│   ├── prompts.py
│   └── retriever.py
└── tests
    ├── test_generator.py
    ├── test_ingest.py
    └── test_retriever.py
```

## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd RAG-Flash-card-app
    ```

2. **Set up the development container:**
    - Open the project in Visual Studio Code.
    - Use the Dev Containers extension to open the project in a container.

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    - Create a [.env](http://_vscodecontentref_/6) file in the root directory and add your environment variables.

## Usage

1. **Run the application:**
    ```sh
    streamlit run src/app.py
    ```

    The app will be running at localhost:8501

2. **Generate flashcards:**
    - Use the [generate_flashcard](http://_vscodecontentref_/7) function in [generator.py](http://_vscodecontentref_/8) to create flashcards from a given context.

3. **Question answering:**
    - Use the [generate_flashcard](http://_vscodecontentref_/7) function in [generator.py](http://_vscodecontentref_/8) to give answer from a relevant context.

## Project Components
- **Data:** Contains PDF files used for generating flashcards.
- **DB:** Contains the database files.
- **Models:** Contains the embeddings and other model-related files.
- **Notebooks:** Contains Jupyter notebooks for exploration and analysis.
- **Source (src):** Contains the main application code.
  - **app.py:** Main application entry point.
  - **generator.py:** Contains functions for generating flashcards.
  - **ingest.py:** Contains functions for ingesting data.
  - **prompts.py:** Contains prompt templates.
  - **retriever.py:** Contains functions for retrieving data.
- **Tests:** Contains unit tests for the application.


⭐️ If you find this repository helpful, please consider giving it a star!

Keywords: RAG, Retrieval-Augmented Generation, NLP, AI, Machine Learning, Information Retrieval, Natural Language Processing, LLM, Embeddings