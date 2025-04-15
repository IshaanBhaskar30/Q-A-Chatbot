import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os

# Title of the app
st.title("Enhanced Q&A Chatbot With Ollama")

# Sidebar API Key input
user_api_key = st.sidebar.text_input("Enter your Langchain API Key", type="password")

# Project name input
user_project_name = st.sidebar.text_input("LangChain Project Name", value="Q&A Chatbot With Ollama")

# Sidebar model and parameters
llm = st.sidebar.selectbox("Select Open Source Model", ["gemma2:2b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Check if API key is provided
if user_api_key:
    os.environ["LANGCHAIN_API_KEY"] = user_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = user_project_name

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user's queries."),
            ("user", "Question: {question}")
        ]
    )

    def generate_response(question, engine, temperature, max_tokens):
        llm_model = Ollama(model=engine)
        output_parser = StrOutputParser()
        chain = prompt|llm_model|output_parser
        answer = chain.invoke({'question': question})
        return answer

    st.write("Go ahead and ask any question:")
    user_input = st.text_input("You:")

    if user_input:
        response = generate_response(user_input, llm, temperature, max_tokens)
        st.write(response)
    else:
        st.write("Please enter a question above.")

else:
    st.warning("Please enter your LangChain API Key in the sidebar to proceed.")
