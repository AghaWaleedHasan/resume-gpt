import sys
sys.dont_write_bytecode = True
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import streamlit as st

class ChatBot():

    def __init__(self, type: str = "openai"):
        self.llm = ChatOpenAI(
            # model='gpt-3.5-turbo',
            model='gpt-4o',
            api_key=st.secrets.openai_key,
            temperature=0.1
        )

    def generate_message_stream(self, question: str, docs: list, history: list, type: str):
        context = ""
        
        if type == "1":
            for i in range(len(docs)):
                pre_system_message = SystemMessage(content="""
                You are given text from a resume and a question. Analyse the resume and determine if the resume is relevant at all to the question.
                If the resume doesn't contain any mention of the requirements placed in the question, reply with just NO. Otherwise, reply with just YES.
                """)

                pre_human_message = HumanMessage(content=f"""
                Question: {question}                                                                
                \nResume: {docs[i]}
                """)

                answer = self.llm.invoke([pre_system_message, pre_human_message])
                # print("\n\n\n"+" ".join(docs[i].split(" ")[:4]))
                # print(answer)

                if (answer.content == "YES"):
                    context += "\n\n"+docs[i]

            # print("context")
            system_message = SystemMessage(content="""
                You are a talent acquisition assistant.
                Use the resumes given in the context and summarize EACH ONE of them in your answer and answer the specific question given by the user. 
                If there are no resumes in the context, reply with 'There were no resumes found relevant to your query'.
            """)

            user_message = HumanMessage(content=f"""
                Context: {context}
                Question: {question}
            """)

        else:
            system_message = SystemMessage(content="""
                You are an expert in talent acquisition that helps analyze resumes to assist resume screening.
                You may use the following pieces of context and chat history to answer your question. 
                Do not mention in your response that you are provided with a chat history.
                If you don't know the answer, just say that you don't know, do not try to make up an answer.
            """)

            user_message = HumanMessage(content=f"""
                Chat history: {history}
                Question: {question}
                Context: {context}
            """)

        stream = self.llm.stream([system_message, user_message])

        return stream

    def query_classification(self, question: str):
        system_message = SystemMessage(content="""
            Classify the user's prompt into one of the following two categories: 
                1. Request for matching profiles or resumes with a given criteria
                2. Other
            Do not include any character or punctuation in your answer. Only respond with a number to indicate your classification. 
        """)

        user_message = HumanMessage(content=f"""
            Prompt: {question}
        """)

        response = self.llm.invoke([system_message, user_message])
        return response.content[0]
