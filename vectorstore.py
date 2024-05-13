from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import streamlit as st
import os

os.environ['OPENAI_API_KEY'] = st.secrets.openai_key

FAISS_PATH = "./vectorstore"

documents = []

resume_paths = ['data/AliAsgharAamir_CV_Amazon.pdf',
             'data/Syed Ali Hussain Resume.pdf',
             'data/Zaid\'s Resume.pdf',
                'data/662c373bdddcfda377bf29fd_Laila_Dodhy_Resume.pdf',
                'data/Hassan CV.pdf',
                'data/CV - Waasif.pdf',
                'data/Imtisal_CV_traton.pdf',
                'data/Mubashir_s_CV.pdf',
                'data/Resume - Agha Waleed Hasan.pdf',
                'data/Usman_Adil_CV.pdf',
                'data/ali_resume_march.pdf',
                'data/MUHAMMAD SHERAZ CV.pdf',
                'data/Hassan-Raza-Resume.pdf',
                'data/Taimoor Arif - Resume.pdf',
                'data/TehreemArshad_Resume.pdf']

for doc_path in resume_paths:
  loader = UnstructuredPDFLoader(doc_path, mode="single", strategy="fast") 
  documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=256)
splits = text_splitter.split_documents(documents)


vectorstore = FAISS.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)

vectorstore.save_local(FAISS_PATH)