import streamlit as st
from langchain_google_genai import GoogleGenerativeAI

st.title("Chatbot")

llm = GoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=st.secrets['GOOGLE_API_KEY'])

#helper function to display and send streamlit messages
def llm_function(bin_switch:int,query:str):

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
    initial_prompt = "Introduce yourself as Dox, an assistant powered by Google Gemini. You use emojis to be interactive and be funny."   
    message = llm.invoke(initial_prompt)
    llm_function(1,message) 

if prompt := st.chat_input("What's up?"):
    llm_function(0,prompt)
    message = llm.invoke(prompt)
    llm_function(1,message)