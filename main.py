import os
from os.path import isfile, join

from PyPDF2 import PdfReader
from langchain.text_splitter import  CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAIChat

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") =="":
        print("OPENAI_API_KEY is not set")
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Peeb GPT", page_icon=":bookmark_tabs:")



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text)
    return  text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = OpenAIChat()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def train_my_pdfs():
    mypath = os.getcwd() + "/training_docs/"
    pdf_files = [(mypath+f) for f in os.listdir(mypath) if isfile(join(mypath, f))]
    print("pdf files == ", pdf_files)
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)


def main():
    init()


    print("hello world!")

    chat = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)
    print(chat.model_name)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]



    st.header("Your Private Tax GPT :male-judge: :money_with_wings:")

    # message(("Hello! How are you?"))
    # message("I'm good", is_user=True)

    with st.sidebar:
        # st.subheader("Your documents")
        # pdf_docs = st.file_uploader(
        #     "Upload your PDFs here and click on Process", accept_multiple_files=True)
        #
        # if st.button("Process"):
        #     with st.spinner("Processing.."):
        #         # get pdf text
        #         raw_text = get_pdf_text(pdf_docs)
        #         # st.write(raw_text)
        #
        #         # get the text chunks
        #         text_chunks = get_text_chunks(raw_text)
        #         # st.write(text_chunks)
        #
        #         # create the vector store
        #         vectorstore = get_vectorstore(text_chunks)
        #
        #         # create conversation chain
        #         st.session_state.conversation = get_conversation_chain(vectorstore)

        if st.button("Process"):
            with st.spinner("Processing.."):
                train_my_pdfs()
        user_input = st.text_input("Your Tax Question? :male-teacher:", key="user_input")



        if user_input:
            #message(user_input, is_user=True)
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
            #message(response.content, is_user=False)

    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i%2 ==0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main()