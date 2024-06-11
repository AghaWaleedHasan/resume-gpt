import os
import re
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

os.environ['OPENAI_API_KEY'] = st.secrets.openai_key

FAISS_PATH = "./vectorstore"

def get_file_paths(directory):
    file_paths = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the full file path and add it to the list
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths

# Specify the directory you want to search
directory_to_search = './data'

# Get the list of file paths
resume_paths = get_file_paths(directory_to_search)

# print("resume paths: ", resume_paths)

documents = []

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^\w\s\d]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back to string
    return ' '.join(tokens)

for doc_path in resume_paths:
    # print(doc_path)
    loader = UnstructuredPDFLoader(doc_path, mode="single", strategy="fast") 
    raw_documents = loader.load()
    for doc in raw_documents:
        cleaned_text = clean_text(doc.page_content)
        doc.page_content = cleaned_text
        documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=128,
    chunk_overlap=32)
splits = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)

vectorstore.save_local(FAISS_PATH)

# bm25_retriever = BM25Retriever.from_documents(splits)
# bm25_retriever.k = 5

# ensemble_retriever = EnsembleRetriever


# from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores.faiss import DistanceStrategy
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import UnstructuredPDFLoader
# import streamlit as st
# import os

# os.environ['OPENAI_API_KEY'] = st.secrets.openai_key

# FAISS_PATH = "./vectorstore"

# documents = []

# resume_paths = ['data/AliAsgharAamir_CV_Amazon.pdf',
#              'data/Syed Ali Hussain Resume.pdf',
#              'data/Zaid\'s Resume.pdf',
#                 'data/662c373bdddcfda377bf29fd_Laila_Dodhy_Resume.pdf',
#                 'data/Hassan CV.pdf',
#                 'data/CV - Waasif.pdf',
#                 'data/Imtisal_CV_traton.pdf',
#                 'data/Mubashir_s_CV.pdf',
#                 'data/Resume - Agha Waleed Hasan.pdf',
#                 'data/Usman_Adil_CV.pdf',
#                 'data/ali_resume_march.pdf',
#                 'data/MUHAMMAD SHERAZ CV.pdf',
#                 'data/Hassan-Raza-Resume.pdf',
#                 'data/Taimoor Arif - Resume.pdf',
#                 'data/TehreemArshad_Resume.pdf']

# for doc_path in resume_paths:
#   loader = UnstructuredPDFLoader(doc_path, mode="single", strategy="fast") 
#   documents.extend(loader.load())

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=300,
#     chunk_overlap=50)
# splits = text_splitter.split_documents(documents)


# vectorstore = FAISS.from_documents(documents=splits,
#                                     embedding=OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)

# vectorstore.save_local(FAISS_PATH)