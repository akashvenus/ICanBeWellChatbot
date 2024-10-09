# ICanBeWellChatbot

[![Python](https://img.shields.io/badge/-Python%203.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Database](https://img.shields.io/badge/database-Chroma%20DB-blue.svg)](https://docs.chromadb.com/)
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

- Create your .env file and store your hugging face api and Google api. The file should look like this
```mardown
HUGGINGFACEHUB_API_TOKEN = <your_api_key>
GOOGLE_API_KEY = <your_api_key>
```
**Running**
- First run rag_model.py
```markdown
python rag_model.py
```
- Now run the chatbot.py
```markdown
streamlit run chatbot.py
```

**License:**

This project is licensed under the MIT License - see the `LICENSE` file for details.
