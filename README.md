# ICanBeWellChatbot

[![Python](https://img.shields.io/badge/-Python%203.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Database](https://img.shields.io/badge/database-Milvus%20DB-blue.svg)](https://milvus.io/docs)
[![Docker](https://img.shields.io/badge/-Docker-384d54?logo=docker)](https://docs.docker.com/)
[![UI](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A chatbot that is used to help users to communicate their problems with the ICanBeWell portal

**Setup**
- Create a virtual environment using python
```markdown
python -m venv <environment_name>
```

- pip install all the requirements
```markdown
pip install -r requirements.txt
```

- Create your .env file and store your secrets. The file should look like this
```mardown
HUGGINGFACEHUB_API_TOKEN = <your_api_key>
GOOGLE_API_KEY = <your_api_key>
COLLECTION_NAME="medical_kb"
MILVUS_HOST="localhost"
MILVUS_PORT="19530"
LANGCHAIN_API_KEY= <your_api_key>
```
**Running**
- First build the docker file using docker-compose up. This will create a docker container with three sub-containers. Use your Docker Desktop to view the containers
  ```markdown
  docker-compose up
  ```
- With the docker containers running in the background you'll have to run rag_model.py
```markdown
python rag_model.py
```
- Now run the chatbot.py with the docker containers running in the background
```markdown
streamlit run chatbot.py
```

**License:**

This project is licensed under the MIT License - see the `LICENSE` file for details.
