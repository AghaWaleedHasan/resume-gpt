import streamlit as st 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.vectorstores import FAISS
from rag_system import RAGPipeline
import streamlit as st
from llm import ChatBot
import os
import openai
import retriever_report

openai.api_key = st.secrets.openai_key
os.environ['OPENAI_API_KEY'] = st.secrets.openai_key

st.header("Resume Screening GPT 💬 📚")

welcome_message = """
#### Introduction 🚀

The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of hundreds of resumes more effectively. ⚡

The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

#### Getting started 🛠️

1. To set up, please add your OpenAI's API key. 🔑 
2. Type in a job description query. 💬

"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

with st.sidebar:
    st.markdown("# Control Panel")
    st.text_input("OpenAI's API Key", type="password", key="api_key")
    st.button("Clear conversation", on_click=clear_message)

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            if not isinstance(message, tuple):
                st.markdown(message.content)

user_query = st.chat_input("Your message")

if "rag_pipeline" not in st.session_state:
    vectordb = FAISS.load_local("./vectorstore", OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    st.session_state.rag_pipeline = RAGPipeline(vectordb)

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []

llm = ChatBot()

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        classification = llm.query_classification(user_query)
        if classification == "1":
            with st.spinner("Generating Answers..."):
                doc_id_with_score = st.session_state.rag_pipeline.retrieve_id_and_rerank(user_query)
                if len(doc_id_with_score) == 0:
                    response = "No relevant resumes found matching the given criteria."
                else:
                    retrieved_docs = st.session_state.rag_pipeline.retrieve_documents_with_id(doc_id_with_score)
                    st.session_state.resume_list = retrieved_docs
                    stream = llm.generate_message_stream(user_query, st.session_state.resume_list, st.session_state.chat_history, classification)
                    response = st.write_stream(stream)
                    retriever_message = retriever_report
                    retriever_message.render(retrieved_docs, doc_id_with_score)
                    st.session_state.chat_history.append(AIMessage(content=response))
        else:
            stream_message = llm.generate_message_stream(user_query, st.session_state.resume_list, st.session_state.chat_history, classification)
            response = st.write_stream(stream_message)
            st.session_state.chat_history.append(AIMessage(content=response))

