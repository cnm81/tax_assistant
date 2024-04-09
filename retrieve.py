from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader  # For loading PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st

import os

openai_api_key = st.secrets["openai_api_key"]
#TMP
#load_dotenv()
#openai_api_key = os.getenv('OPENAI_API_KEY')

def load_db(index_file='vectorstore'): 
    emb = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(index_file, emb, allow_dangerous_deserialization=True)

def chat_with_documents(query, retriever, n_results=5):
    # Use the retriever to find the most relevant document chunk
    results = retriever.get_relevant_documents(query, n_results=n_results)
    if not results:
        print("No relevant information found.")
        return
    context = " ".join([result.page_content for result in results]) 
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    prompt = f"Answer the question: {query} - \nGiven the following information: {context}\n\n" 
    messages = [
        SystemMessage(
            content="You are a helpful assistant that Explains tax questions. Use only the relevant part of the information given to answer the question."
        ),
        HumanMessage(
            content=prompt
        ),
    ]
    response = chat.invoke(messages, max_tokens=100)

    return response.content


if __name__ == "__main__":
    # Example usage

    db = load_db()

    query = "Hur mycket behöver jag betala i arbetsgivaravgift? "

    # Initialize the retriever
    retriever = db.as_retriever()
    response = chat_with_documents(query, retriever)
    print(f"Answer: {response}")
