# RAG-Based Medical Chatbot

A Streamlit medical reference chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from indexed medical documents.

The app loads medical PDF files, converts them into text chunks, stores those chunks in a FAISS vector database, and uses a Groq-hosted LLM with LangChain to answer user questions with source-backed context.

## Features

- RAG pipeline for document-grounded medical question answering
- FAISS vector database for semantic search
- HuggingFace sentence-transformer embeddings
- Groq LLM integration through `langchain-groq`
- Streamlit chat interface
- Source citation previews for retrieved document chunks
- Adjustable model, temperature, and retrieval count from the sidebar

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API
- dotenv

## Project Structure

```text
.
|-- data/
|   `-- The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
|-- vectorstore/
|   `-- db_faiss/
|-- connect_memory_with_llm.py
|-- create_memory_for_llm.py
|-- main.py
|-- medical_bot.py
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Clone the Project

```bash
git clone https://github.com/nitin08240/RAG--BASED-MEDICAL-CHATBOT.git
cd RAG--BASED-MEDICAL-CHATBOT
```

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

5. Add your medical PDF files to the `data/` folder.

6. Build the FAISS vector database:

```bash
python create_memory_for_llm.py
```

This creates the local vector database at:

```text
vectorstore/db_faiss
```

7. Run the Streamlit app:

```bash
streamlit run medical_bot.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. PDFs inside `data/` are loaded with LangChain document loaders.
2. The extracted text is split into smaller overlapping chunks.
3. Chunks are converted into embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. Embeddings are stored locally in a FAISS vector database.
5. When a user asks a question, the app retrieves the most relevant chunks.
6. The retrieved context is passed to the Groq LLM to generate a grounded answer.

## Important Notes

- This project is for educational and reference purposes only.
- It does not replace professional medical advice, diagnosis, or treatment.
- Always verify medical information with qualified healthcare professionals and current clinical guidelines.
- Keep your `.env` file private. Do not commit API keys to GitHub.

## Useful Commands

```bash
# Rebuild vector database after adding or changing PDFs
python create_memory_for_llm.py

# Run the chatbot
streamlit run medical_bot.py

# Check Git status
git status
```

## Repository

GitHub: https://github.com/nitin08240/RAG--BASED-MEDICAL-CHATBOT
