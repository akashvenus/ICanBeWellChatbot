from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from chatbot import llm, full_prompt as user_input

def load_data():
    loader = JSONLoader(
        file_path='data/Aorta-EN.json',
        jq_schema='.',
        text_content=False)
    return loader.load()

def preprocessing(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=35)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def load_vector_store(embeddings,text_splits):
    vector_store = Milvus.from_documents(
            embedding=embeddings,
            documents=text_splits,
            connection_args={
                "uri": "./milvus_demo.db",
            },
            drop_old=True
        )
    return vector_store


prompt = PromptTemplate.from_template(
    """I want you to analyze the provided medical details of a few patients within the {context}. Focus on the following keys and their descriptions:
    - Topic heading: Type of ailment
    - Gender: Gender of patient
    - General Patient Text: Preventative measures for the ailment (patient-specific advice)
    - Health Provider Text: Preventative measures for the ailment from external sources (general advice)
    - Subject: Description of patient

    Carefully consider the user's specific details provided in the {user_input} (age, symptoms, etc.) and tailor your response accordingly.
    Synthesize information from the relevant keys to answer the {user_input}. Prioritize patient-specific advice from "General Patient Text." If "General Patient Text" is "n/a," provide the external link in "Health Provider Text" while acknowledging the absence of patient-specific advice. If "Health Provider Text" is "n/a", rely solely on "General Patient Text."
    Provide your answer in a concise paragraph, addressing the user's query directly. Additionally, if the user's question pertains to preventative measures or external resources, extract and display any relevant links found within "General Patient Text" or "Health Provider Text." Present the links clearly and indicate their source (e.g., "Link from General Patient Text: [link]")
    Present the links using Markdown formatting to make them clickable. For example, use (link URL) to create a clickable link.
    """
)

data = load_data()

data_splits = preprocessing(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

retriever = load_vector_store(embeddings,data_splits).as_retriever()

# input_question = "I am a 70 years old man and I believe i have an issue with my aorta. Do i need an ultrasound ?"

for item in user_input:
    print(item)

# rag_chain = (
#     {"context": retriever, "user_input": user_input}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# response = rag_chain.invoke()
