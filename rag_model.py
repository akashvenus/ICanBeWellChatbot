from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import json
from collections import defaultdict
 
# Load environment variables from .env file
load_dotenv()
 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
 
loader = JSONLoader(
        file_path='data/HtmlTopic-EN-final.json',
        jq_schema='.[] | {button: .Button, topic_heading: ."Topic heading", subject: .Subject, general_patient_text: ."General Patient Text", health_provider_text: ."Health Provider Text", gender: .Gender, min_age: ."Minimum age", max_age: ."Maximum age"}',
        text_content=False)
docs = loader.load()

for doc in docs:
    page_content = json.loads(doc.page_content)
    doc.metadata.update({
    'button': page_content['button'],
    'topic_heading': page_content['topic_heading'],
    'gender': page_content['gender'],
    'min_age': page_content['min_age'],
    'max_age': page_content['max_age']
        })
    del page_content['button']
    del page_content['topic_heading']
    del page_content['gender']
    del page_content['min_age']
    del page_content['max_age']
    doc.page_content = json.dumps(page_content)
# print(docs)
# print(docs)
 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(docs) 
 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# doc_groups = defaultdict(list)
# for doc in docs:
#     topic_heading = doc.metadata['topic_heading']
#     doc_groups[topic_heading].append(doc)


# print(doc_groups)

# Create separate vector stores for each topic heading
# for topic_heading, topic_docs in doc_groups.items():
#     topic_store = Chroma(embedding_function=embedding_model, persist_directory=f"data/{topic_heading}")
#     topic_store.add_documents(topic_docs)
 
 
# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data/Aorta")
 
# vector_store.add_documents(documents=texts) #Correction
 
 
# Test query to validate data retrieval
test_query = "I am 70 years old and I believe i have an issue with my aorta. Do i need an ultrasound?"
results = vector_store.search(query=test_query, search_type='similarity')
print(results)