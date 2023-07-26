import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import base64

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf):
    try:
        pdf_reader = PdfReader(pdf)
        return ''.join(page.extract_text() for page in pdf_reader.pages), None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
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
        return response, None
    except Exception as e:
        return None, str(e)

def display_pdf(pdf):
    base64_pdf = base64.b64encode(pdf.getvalue()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="400" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Ravi PDF Reader")
    st.header("Chat with your PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        text, error = extract_text_from_pdf(pdf)
        if error:
            st.error(f"Failed to read the PDF. Error: {error}")
        elif text:
            openai_api_key = st.secrets['openai']["OPENAI_API_KEY"]
            knowledge_base = generate_embeddings(text, openai_api_key)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Display PDF in the first column
            col1.subheader("PDF Content")
            display_pdf(pdf)

            # Allow user to ask questions in the second column
            col2.subheader("Ask a question about your PDF:")
            user_question = col2.text_input("")
            if user_question:
                response, error = get_answer_for_question(knowledge_base, openai_api_key, user_question)
                if error:
                    col2.error(f"Error fetching the answer: {error}")
                elif response:
                    col2.write(f"From PDF: {response}")

if __name__ == '__main__':
    main()
