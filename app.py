import streamlit as st
from PyPDF2 import PdfFileReader, PdfFileError
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

@st.cache(show_spinner=False)  # Cache to avoid repeated computations
def extract_text_from_pdf(pdf):
    try:
        pdf_reader = PdfFileReader(pdf)
        return ''.join(page.extract_text() for page in pdf_reader.pages)
    except PdfFileError:
        st.error("Failed to read the PDF. Please upload a valid PDF.")
        return None

@st.cache(show_spinner=False)
def generate_embeddings(text, openai_api_key):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_texts(chunks, embeddings)

def get_answer_for_question(knowledge_base, openai_api_key, user_question):
    try:
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI(openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
        return response
    except Exception as e:
        st.error(f"Error fetching the answer: {str(e)}")
        return None

def ask_chatgpt(user_question):
    # Placeholder: Add functionality to call ChatGPT here and get a response
    return f"ChatGPT says: {user_question}"

def main():
    st.set_page_config(page_title="Ravi PDF Reader")
    st.header("Chat with your PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        text = extract_text_from_pdf(pdf)
        if text:
            openai_api_key = st.secrets['openai']["OPENAI_API_KEY"]
            knowledge_base = generate_embeddings(text, openai_api_key)
            
            # Create two columns
            col1, col2 = st.beta_columns(2)
            
            # Display PDF content in the first column
            col1.subheader("PDF Content")
            col1.write(text)

            # Allow user to ask questions in the second column
            col2.subheader("Ask a question about your PDF:")
            user_question = col2.text_input("")
            if user_question:
                response = get_answer_for_question(knowledge_base, openai_api_key, user_question)
                if response:
                    col2.write(f"From PDF: {response}")
                    # Optionally, also ask ChatGPT for supplementary info
                    chatgpt_response = ask_chatgpt(user_question)
                    col2.write(chatgpt_response)

if __name__ == '__main__':
    main()
