from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus  
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import os
from collections import OrderedDict
import json
from dotenv import load_dotenv
from pymilvus import connections, utility


load_dotenv()


os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Milvus configuration
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

def setup_milvus():
    """Setup Milvus collection"""
    try:
        # Check if collection exists and delete if it does
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
            print(f"Dropped existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error setting up Milvus: {e}")
        raise

try:
    # Connect to Milvus first
    connect_to_milvus()
    

    # Load the document
    loader = JSONLoader(
        file_path='data/HtmlTopic-EN-final.json',
        jq_schema='.[] | {button: .Button, topic_heading: ."Topic heading", subject: .Subject, general_patient_text: ."General Patient Text", health_provider_text: ."Health Provider Text", gender: .Gender, min_age: ."Minimum age", max_age: ."Maximum age"}',
        text_content=False)
    docs = loader.load()

    for doc in docs:
        page_content = json.loads(doc.page_content)

        # Directly update metadata and remove keys in one pass
        for key in ['button', 'topic_heading', 'gender', 'min_age', 'max_age']:
            doc.metadata[key] = page_content.pop(key)

        doc.page_content = json.dumps(page_content)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs) 
     
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # Setup Milvus collection
    setup_milvus()
    
    # Initialize vector store with the new Milvus class
    vector_store = Milvus(
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT
        },
        auto_id=True
    )
    
    # Add documents in smaller batches
    batch_size = 50  # Reduced batch size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # print(batch)
        vector_store.add_documents(documents=batch)
        print(f"Added batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")
    
    print("Documents added to Milvus successfully")

    # Test query
    test_query = "I am 70 years old and I believe i have an issue with my aorta. Do i need an ultrasound?"
    results = vector_store.similarity_search(test_query)

    # Deduplicate results
    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc

    # Convert unique results to a list and limit to top 3
    final_results = list(unique_results.values())[:3]
    print(f"Unique query results: {final_results}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Always disconnect from Milvus
    connections.disconnect("default")

