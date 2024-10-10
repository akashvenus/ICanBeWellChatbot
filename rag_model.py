from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


loader = JSONLoader(
        file_path='data/Topics-EN.json',
        jq_schema='.',
        text_content=False)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=35)
texts = text_splitter.split_documents(docs)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert texts to embeddings
try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
    print("Vector Embeddings created successfully")
except Exception as e:
    print(f"Error creating vector embeddings: {e}")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")

# Add documents to the vector store
vector_store.add_documents(documents=texts)

# Validate the setup
try:
    # Test query to validate data retrieval
    test_query = "I am 70 years old and I believe i have an issue with my aorta. Do i need an ultrasound ?"
    results = vector_store.search(query=test_query, search_type='similarity')

    # Deduplicate results
    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc

    # Convert unique results to a list and limit to top 3
    final_results = list(unique_results.values())[:3]
    print(f"Unique query results: {final_results}")
except Exception as e:
    print(f"Error during test query: {e}")

