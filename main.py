import os

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

def main():
    init()

    print("hello world!")

    chat = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)
    print(chat.model_name)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]



    st.header("Peeb Tax GPT :male-judge: :money_with_wings:")

    # message(("Hello! How are you?"))
    # message("I'm good", is_user=True)

    with st.sidebar:
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