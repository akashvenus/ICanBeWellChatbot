import streamlit as st
# from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import re

st.title("Chatbot")

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

llm = genai.GenerativeModel("gemini-1.5-flash")


#helper function to display and send streamlit messages
def display_message(bin_switch:int,query:str):

    if bin_switch % 2 == 0:
        st.session_state.messages.append({"role":"user","content":query})
        with st.chat_message("user"):
            st.markdown(query)
    
    else:
        with st.chat_message("bot"):
            st.markdown(query)
            st.session_state.messages.append({"role":"bot","content":query})
    
    return

if "messages" not in st.session_state:
    st.session_state.messages = []



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    initial_prompt = '''Introduce yourself as Dox, an assistant powered by Google Gemini. You have to collect only the following information 
    from the user initially:
    - Age
    - Current Gender (Male, Female or Non-binary)
    - Gender assigned at birth (Male or Female)
    - Whether they are member of public or a health professional

    If the user does not give the above information or asks anything irrelevant, refuse politely and ask for the details again.
    '''  
    message = llm.generate_content(initial_prompt)
    display_message(1,message.candidates[0].content.parts[0].text) 

if prompt := st.chat_input("What's up?"):
    display_message(0,prompt)
    
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    full_prompt = conversation_history + "\nuser: " + prompt

    message = llm.generate_content(full_prompt)
    display_message(1,message.candidates[0].content.parts[0].text)