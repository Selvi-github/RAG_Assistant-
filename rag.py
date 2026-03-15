import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv 

# Load environment variables at the top level
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files with extensive error handling."""
    text = ""
    for pdf in pdf_docs:
        try:
            # Basic validation: Check if file is small or has invalid header
            content = pdf.read(10)
            pdf.seek(0)
            if content.startswith(b"PK"):
                st.warning(f"⚠️ {pdf.name} appears to be a ZIP or Word (.docx) file renamed to .pdf. Please upload a real PDF.")
                continue
            
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            if "EOF marker not found" in str(e):
                st.error(f"❌ {pdf.name} is corrupt or not a valid PDF file (EOF marker missing).")
            else:
                st.error(f"❌ Error reading {pdf.name}: {e}")
            continue
    return text

def get_text_chunks(text):
    """Splits text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found. Check your .env file.")
            return

        # Using models/gemini-embedding-001 for stability
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False

def user_input(user_question):
    """Handles user query by searching vector store and calling the model directly."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        
        if not os.path.exists("faiss_index"):
            st.warning("No document index found. Please upload and process documents first.")
            return

        with st.status("🔍 Analyzing documents...", expanded=True) as status:
            st.write("Searching database...")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(user_question)
            
            if not docs:
                status.update(label="No matches found.", state="error")
                st.warning("No relevant information found in the documents.")
                return

            st.write("Preparing context...")
            context = "\n".join([doc.page_content for doc in docs])
            
            prompt_template = """
            Answer the question as detailed as possible from the provided context and with your own knowledge.
            If the answer is not in the provided context, try to give the correct answer with your knowledge, 
            but make sure you don't give any wrong information.
            
            Context:\n {context}\n
            Question:\n {question}\n
            Answer:
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            final_prompt = prompt.format(context=context, question=user_question)
            
            st.write("Invoking Gemini...")
            # Using models/gemini-flash-latest for best performance and availability
            model = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.3, google_api_key=api_key)
            
            response = model.invoke(final_prompt)
            
            # Newer Gemini models return content as a list of blocks, not a plain string
            if response and response.content:
                if isinstance(response.content, list):
                    # Extract text from each block in the list
                    answer = "\n".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in response.content
                    ).strip()
                else:
                    answer = str(response.content).strip()
                
                if answer:
                    status.update(label="Answer generated!", state="complete")
                    st.write("### Response:")
                    st.success(answer)
                else:
                    status.update(label="No answer generated.", state="error")
                    st.warning("The AI could not generate a response.")
            else:
                status.update(label="No answer generated.", state="error")
                st.warning("The AI returned an empty response.")
                
    except Exception as e:
        st.error(f"Search flow failed: {e}")

def main():
    st.set_page_config(page_title="Pro RAG Assistant", layout="wide", page_icon="🤖")
    
    # Force dark theme style injection
    st.markdown("""
        <style>
            .stApp {
                background-color: #0a0a0a !important;
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] {
                background-color: #1a1a1a !important;
                border-right: 1px solid #333333;
            }
            .stTextInput>div>div>input {
                color: #ffffff !important;
                background-color: #1a1a1a !important;
                border: 1px solid #333333 !important;
            }
            .stButton>button {
                background-color: #00d1ff !important;
                color: #0a0a0a !important;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Pro RAG Assistant 📚")
    st.markdown("---")
    
    # Initialization
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"API Configuration failed: {e}")
    else:
        st.warning("⚠️ GOOGLE_API_KEY is missing in .env file.")

    user_question = st.text_input("Ask a Question about your documents", placeholder="e.g., What are the key findings in these files?")
    if user_question:
        if api_key:
            user_input(user_question)
        else:
            st.error("API Key is required to chat.")

    with st.sidebar:
        st.header("Upload Center")
        pdf_docs = st.file_uploader("Drop your PDF files here", accept_multiple_files=True, type=["pdf"])
        
        if st.button("🚀 Process Documents"):
            if not pdf_docs:
                st.info("No files selected.")
            elif not api_key:
                st.error("Missing API Key.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if get_vector_store(text_chunks):
                            st.success("Indexing complete! Ask away 💬")
                        else:
                            st.error("Failed to build index.")
                    else:
                        st.error("No valid text could be extracted.")

if __name__ == "__main__":
    main()
