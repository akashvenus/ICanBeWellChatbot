import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from pymilvus import connections
# from langchain_chroma import Chroma
from pymilvus import connections
from langchain_milvus import Milvus
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from rag_model import connect_to_milvus
import json

load_dotenv()

# data_directory = os.path.join(os.path.dirname(__file__), "data")


#For Tracing and logging
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="ICanBeWellChatbot"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("Chatbot")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_kb")

def connect_to_milvus():
    """Establish connection to Milvus"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print("Successfully connected to Milvus!")
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        raise

connect_to_milvus()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Milvus(
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT
        },
        auto_id=True
)
#LLM
llm = GoogleGenerativeAI(model='gemini-1.5-flash',api_key=os.getenv('GOOGLE_API_KEY'), temperature=0)

prompt = PromptTemplate.from_template(
        """
        I want you to respond only according to the following information given below:
        
       [1] I want you to fetch information from the given database {database}. The keys in the database represent the following information:
        - Topic heading: Type of ailment
        - Gender: Gender of patient. Below is the mapping for the genders.
		  m -> male or man
		  f -> female or woman
		  tf -> trans-female or trans female or transfeminine
		  tm -> trans-male or trans male or transmasculine
	      all -> any gender
        - General Patient Text: Patient specific advice for the users who are 'Member of public'
        - Health Provider Text: Patient specific advice for the users who are 'Health Professional'
        - Subject: Description of the information present in 'General Patient Text' and 'Health Provider Text'
        - Minimum age: Minimum age of the user/patient 
        - Maximum age: Maximum age of the user/patient

        [2] There are two user types: 'Member of the public' and 'Health Professional'.   
        If the 'Gender' allocated for a particular 'Subject' is 'all' then refrain asking gender information from the user. If the user's age falls below the 18, kindly say that there is no information to be provided.
        If the user is a 'Member of public' then prioritize patient-specific advice from "General Patient Text". If the user is a 'Health Professional' then prioritize patient-specific advice from "Health Provider Text."`
        Provide your answer in a concise paragraph, addressing the user's query directly. Additionally, if the user's question pertains to preventative measures or external resources, extract and display any relevant links found within "General Patient Text" or "Health Provider Text".
        
        Carefully consider the user's specific details provided in the {user_input} (age, gender, user type, subject, etc.) and tailor your response accordingly.
        If the user does not specify age, gender, and the user type then based on the information given in point [2] check if the specified inputs are necessary. If so, kindly ask the user for the age, gender, the user type.  
        Understand and distinguish what type of ailment is the user talking about and synthesize information from the relevant keys to answer the {user_input}. 

        [3] You have to keep track of the chat history {chat_history}, remember the conversation and respond accordingly. You should not forget what the user inputted earlier.        

       
        [4] If the user thanks you, say you are welcome and ask them if you can help them with anything else. If they do not want to continue, kindly acknowledge.

        Answer :
        """
    )

# Load symptom keywords and province links files
with open("data\Symptoms.json") as f:
    symptom_data = json.load(f)
    symptom_keywords = set(symptom_data["symptoms"])

with open("data\Province-links.json") as f:
    province_links = json.load(f)

def rag_run(prompt_template,user_input,vector_store,chat_history):

    retriever = vector_store.as_retriever()

    st.write("**Retrieving relevant data from the vector store...**")
    results = retriever.get_relevant_documents(chat_history) 
    
    # Display retrieved documents' metadata
    with st.sidebar:

        st.write("### Documents retrieved for processing:")
        for i, doc in enumerate(results):
            st.markdown(f"**Document {i+1}:**")
            st.write(f"- Subject: {doc.metadata.get('subject', 'N/A')}")
            st.write(f"- Gender: {doc.metadata.get('gender', 'N/A')}")
            st.write(f"- Age Range: {doc.metadata.get('min_age', 'N/A')} to {doc.metadata.get('max_age', 'N/A')}")
            st.write(f"- Content Snippet: {doc.page_content[:350]}...") 

    rag_chain = (
        # {"database": retriever, "user_input": RunnablePassthrough(), "chat_history" : lambda x : chat_history}
        prompt_template
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke({
        "database" : retriever,
        "chat_history" : chat_history,
        "user_input" : user_input
    })
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
    initial_prompt = '''Introduce yourself as Well-bot, an ICanBeWell application's chabot only giving information related to prevention of diseases powered by Google Gemini. Remember, This does not replace the advice of a trained professional. If you have any concerns about your health, please consult with a doctor or other qualified healthcare provider. Call 911 if urgent. '''  
    # message = llm.generate_content(initial_prompt)
    message = llm.invoke(initial_prompt)
    # display_message(1,message.candidates[0].content.parts[0].text) 
    display_message(1,message.content)

def is_symptom_present(user_text):
    return any(symptom in user_text.lower() for symptom in symptom_keywords)

if "expecting_province" not in st.session_state:
    st.session_state.expecting_province = False

if user_text := st.chat_input("What's up?"):
    display_message(0,user_text)
    if st.session_state.expecting_province:
        province = next((prov for prov in province_links if prov.lower() in user_text.lower()), None)
        
        if province:
            # User provided province after being prompted
            province_link = province_links[province]
            symptom_province_instruction = (
                f"Thank you for specifying {province}. Here is the symptom checker link for {province}: "
                f"{province_link}. Remember, this does not replace professional medical advice. If urgent call 911, otherwise call 811."
            )
            display_message(1, symptom_province_instruction)
            # Reset flag as province is now provided
            st.session_state.expecting_province = False
        else:
            # User response didn't include a recognized province
            all_links = "\n".join([f"{prov}: {link}" for prov, link in province_links.items()])
            no_specific_province_instruction = (
                "Not all provinces and territories have a specific symptom checker. You can use a website from another "
                "province, but be aware that the contact information may not apply to you. Here are available options:\n" 
                f"{all_links}. If urgent call 911, otherwise call 811 or go to the 811 website of your province or territory to speak to a nurse."
            )
            display_message(1, no_specific_province_instruction)
    else:
        symptom_flag = is_symptom_present(user_text)
        
        if symptom_flag:
            province = next((prov for prov in province_links if prov.lower() in user_text.lower()), None)
            
            if province:
                province_link = province_links[province]
                symptom_province_instruction = (
                    f"Since you've mentioned symptoms and specified {province}, here is the symptom checker link for {province}: "
                    f"{province_link}. Remember, this does not replace professional medical advice. If urgent call 911, otherwise call 811."
                )
                display_message(1, symptom_province_instruction)
            else:
                symptom_province_instruction = (
                    "Since you've mentioned symptoms, could you let me know which province you are from? "
                    "I'll share a relevant symptom checker link based on your response. Remember, this does not replace "
                    "professional medical advice. If urgent call 911, otherwise call 811."
                )
                display_message(1, symptom_province_instruction)
                st.session_state.expecting_province = True
        else:
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            full_prompt = conversation_history + "\nuser: " + user_text 
            message = rag_run(prompt,user_text,vector_store,full_prompt)
            display_message(1,message)