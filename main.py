import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

if os.getenv("DEPLOYMENT_ENVIRONMENT", "development") != "production":
    load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
openai = OpenAI()

def get_open_ai_client():
    
    # Initializing the OpenAI client
    client = ChatOpenAI(
    model = gpt_model, 
    # Set the seed value to get repeatable/predictable results
    # seed = 100, 

    # Optional setting for temperature; default is 1; temperature
            # can be set up to 2 for more answer randomness/creativity
    temperature = 1, 
    max_tokens = 1000,

    # Optional api_key parameter if you prefer to pass api key in 
    # directly instead of loading environment variables
    # Fetching the OPENAI_API_KEY environment variable from the secrets.toml file
    api_key=openai_api_key,  
    )

    return client


def chat(message, history):
    system_message = "You are a helpful assistant"
    history_langchain_format = [SystemMessage(system_message)]
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = get_open_ai_client().invoke(history_langchain_format)
    return gpt_response.content


view = gr.ChatInterface(
    chat,
    api_name="gradio-chatbot",
)

if __name__== '__main__':
    view.launch(auth=(user, password))