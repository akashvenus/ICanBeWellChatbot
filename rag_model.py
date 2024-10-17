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

# Load Semantics data
# loader_semantics = JSONLoader(
#     file_path='data/Semantics.json',
#     jq_schema='.',
#     text_content=False
# )
# semdocs = loader_semantics.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=35)
texts = text_splitter.split_documents(docs)

#semantics data
# semantics_texts = text_splitter.split_documents(semdocs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert texts to embeddings
try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
    print("Vector Embeddings created successfully")
except Exception as e:
    print(f"Error creating vector embeddings: {e}")

# Convert semantics to embeddings
# try:
#     semantics_embeddings = embedding_model.embed_documents([doc.page_content for doc in semantics_texts])
#     print("Vector embeddings for Semantics created successfully")
# except Exception as e:
#     print(f"Error creating vector embeddings for Semantics: {e}")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")

# Add documents to the vector store
vector_store.add_documents(documents=texts)

# # Initialize Chroma vector store for Semantics
# vector_store_semantics = Chroma(embedding_function=embedding_model, persist_directory="data/semantics")

# # Add documents to the semantics vector store
# vector_store_semantics.add_documents(documents=semantics_texts)

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

    # navigation_query = input("Would you like to know how to navigate within the app? (yes/no): ")
    
    # if navigation_query.lower() == "yes":
    #     # Fetch navigation details using the same query
    #     results_semantics = vector_store_semantics.search(query=test_query, search_type='similarity')

