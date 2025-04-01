import os
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Upload PDF files
st.header("UD Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract text
if file is not None:
    pdf_Reader = PdfReader(file)
    text = ""
    for page in pdf_Reader.pages:
        text += page.extract_text()
        # st.write(text)

# Breaking text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text);
    # st.write(chunks)

    # Generating embeddings

    # Creating vector stores