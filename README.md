# Pro RAG Assistant 🤖

A production-ready RAG (Retrieval-Augmented Generation) application built with **Streamlit**, **LangChain**, and **Google Gemini AI**.

## Features
- 📄 **Upload multiple PDFs** — handles corrupt/invalid files gracefully
- 🔍 **Semantic search** with FAISS vector store
- 🧠 **Google Gemini AI answers** based on your documents
- 🌑 **Premium dark theme** UI

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Selvi-github/RAG_Assistant-
cd RAG_Assistant-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_api_key_here
```
Get your free API key at [Google AI Studio](https://aistudio.google.com/).

### 4. Run the app
```bash
streamlit run rag.py
```

## How to Use
1. Upload PDF files in the **sidebar**.
2. Click **"🚀 Process Documents"** to index them.
3. Type your question in the main text box and get AI-powered answers.

## Tech Stack
| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `pypdf` | PDF text extraction |
| `langchain` | LLM orchestration |
| `langchain-google-genai` | Gemini AI integration |
| `FAISS` | Vector similarity search |
| `python-dotenv` | Environment variable management |
