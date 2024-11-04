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

        [4] If you encounter any conversation where the user discusses their symptoms, for example: "I have a bad back ache" or "I feel like I might catch a cold" to name a few, you should ask them which province they are from. 
        Then based on the map given below, you should share the link based on the province the user inputs. Also, mention that this does not replace the advice of a trained professional and suggest them if urgent to call 911, otherwise call 811. 
        If the province entered by the user is not one of the below, then suggest them to call 911 if urgent. Not all provinces and territories have a symptom checker. The user can use the website of another province, but the contact information may not apply to them. 

        User input Map:
        "Alberta" -> "https://myhealth.alberta.ca/health/Pages/conditions.aspx?hwid=hwsxchk",
        "British Columbia" -> "https://www.healthlinkbc.ca/illnesses-conditions/check-your-symptoms",
        "New Brunswick" -> "https://www2.gnb.ca/content/gnb/en/departments/health/patientinformation/PrimaryHealthCare/What_is_Primary_Health_Care/symptom-checker.html",
        "Ontario" -> "https://health811.ontario.ca/static/guest/symptom-assessment",
        "Saskatchewan" -> "https://www.saskhealthauthority.ca/your-health/conditions-diseases-services/healthline-online/hwsxchk"

        [5] If the user thanks you, say you are welcome and ask them if you can help them with anything else. If they do not want to continue, kindly acknowledge.

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
