import streamlit as st
# import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


st.title("Chatbot")



embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

llm = GoogleGenerativeAI(model='gemini-1.5-flash',api_key=os.getenv('GOOGLE_API_KEY'))

prompt = PromptTemplate.from_template(
        """
        I want you to go respond only according to the following information given below:
        
       [1] I want you to analyze the provided medical details of a few patients from the given database {database}. Focus on the following keys and their descriptions:
        - Topic heading: Type of ailment
        - Gender: Gender of patient
        - General Patient Text: Preventative measures for the ailment (patient-specific advice)
        - Health Provider Text: Preventative measures for the ailment from external sources (general advice)
        - Subject: Description of patient

        [2] Carefully consider the user's specific details provided in the {user_input} (age, symptoms, etc.) and tailor your response accordingly.
        Synthesize information from the relevant keys to answer the {user_input}. Prioritize patient-specific advice from "General Patient Text." If "General Patient Text" is "n/a," provide the external link in "Health Provider Text" while acknowledging the absence of patient-specific advice. If "Health Provider Text" is "n/a", rely solely on "General Patient Text."
        Provide your answer in a concise paragraph, addressing the user's query directly. Additionally, if the user's question pertains to preventative measures or external resources, extract and display any relevant links found within "General Patient Text" or "Health Provider Text." Present the links clearly and indicate their source (e.g., "Link from General Patient Text: [link]")
        Present the links using Markdown formatting to make them clickable. For example, use (link URL) to create a clickable link. 

        [3] You have to keep track of the chat history {chat_history}, remember the conversation and respond accordingly. You should not forget what the user inputted earlier.        

        Answer :
        """
    )

    # 


def rag_run(prompt_template,user_input,vector_store,chat_history):

    retriever = vector_store.as_retriever()

    rag_chain = (
        {"database": retriever, "user_input": RunnablePassthrough(), "chat_history" : lambda x : chat_history}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_input)
    return response

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

# Helper function to summarize conversation
def summarize_conversation(messages):
    # Simple concatenation of the first few messages for summarization
    summary = "Summary: " + " | ".join([msg['content'] for msg in messages[:5]])  
    return summary

def get_user_input_as_dict(user_input):
    return {"user_input": user_input}


if "messages" not in st.session_state:
    st.session_state.messages = []


# Summarize conversation if it exceeds 20 exchanges
if len(st.session_state.messages) > 20:
    summary = summarize_conversation(st.session_state.messages)
    # Replace the first 10 messages with the summary
    st.session_state.messages = [{"role": "bot", "content": summary}] + st.session_state.messages[10:]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    initial_prompt = '''Introduce yourself as Doxy, a medical assistant only giving information related to prevention of diseases powered by Google Gemini. '''  
    # message = llm.generate_content(initial_prompt)
    message = llm.invoke(initial_prompt)
    # display_message(1,message.candidates[0].content.parts[0].text) 
    display_message(1,message)

if user_text := st.chat_input("What's up?"):
    display_message(0,user_text)
    
    #Chat history
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    full_prompt = conversation_history + "\nuser: " + user_text
    # message = llm.generate_content(full_prompt)
    # print(user_text)
    message = rag_run(prompt,user_text,vector_store,full_prompt)
    display_message(1,message)
