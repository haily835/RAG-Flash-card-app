# RAG Flash Card App

This project is a Research Article Generator (RAG) Flash Card App that extracts key points from research articles and generates flashcards with sources and reasoning.

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
    python src/app.py
    ```

2. **Generate flashcards:**
    - Use the [generate_flashcard](http://_vscodecontentref_/7) function in [generator.py](http://_vscodecontentref_/8) to create flashcards from a given context.

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
