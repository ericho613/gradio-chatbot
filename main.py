import os
from dotenv import load_dotenv
# from openai import OpenAI
import gradio as gr
from langchain.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse
import boto3
from botocore.config import Config

if os.getenv("DEPLOYMENT_ENVIRONMENT", "development") != "production":
    load_dotenv(override=True)

# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_base_url = os.getenv("OPENAI_BASE_URL")
gpt_model = os.getenv("GPT_MODEL")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
# openai = OpenAI()
bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
region_name = os.getenv("AWS_REGION")
bedrock_model = os.getenv("AWS_BEDROCK_MODEL")

def get_bedrock_client():

    # Set timeouts in seconds (e.g., 3600 for 1 hour)
    boto_config = Config(
        read_timeout=36000,
        connect_timeout=60,
        retries={"max_attempts": 10}
    )

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=region_name,
        config=boto_config
    )

    # thinking_params= {
    #     "thinking": {"type": "adaptive"},
    #     # effort high is the default
    #     "output_config": {"effort": "max"}
    # }

    # ChatBedrockConverse unifies parameters under Bedrock's Converse API spec
    chat = ChatBedrockConverse(
        bedrock_api_key=bedrock_api_key,
        # region_name=region_name,
        model=bedrock_model,
        temperature=1,
        max_tokens=128000,
        # thinking_params=thinking_params,
        client=bedrock_client
    )

    return chat

# def get_open_ai_client():

    # # Initializing the OpenAI client
    # client = ChatOpenAI(
    #     model=gpt_model,

    #     # Set the seed value to get repeatable/predictable results
    #     # seed=100,

    #     # Optional setting for temperature; default is 1; temperature
    #     # can be set up to 2 for more answer randomness/creativity
    #     temperature=1,
    #     max_tokens=1000,
    #     # max_tokens=128000,

    #     # Optional api_key parameter if you prefer to pass api key in 
    #     # directly instead of loading environment variables
    #     # Fetching the OPENAI_API_KEY environment variable from the secrets.toml file
    #     api_key=openai_api_key,

    #     base_url=openai_base_url
    # )
    # return client


# def read_markdown_files(files):
#     """
#     Reads uploaded markdown files and returns their contents as a single string.
#     """
#     if not files:
#         return ""

#     markdown_content = []
#     for file in files:
#         try:
#             with open(file.name, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 markdown_content.append(
#                     f"\n\n--- Start of file: {os.path.basename(file.name)} ---\n"
#                     f"{content}\n"
#                     f"--- End of file: {os.path.basename(file.name)} ---\n"
#                 )
#         except Exception as e:
#             markdown_content.append(
#                 f"\n\n[Error reading file {os.path.basename(file.name)}: {str(e)}]"
#             )

#     return "\n".join(markdown_content)

# def chat(message, history, files):
def chat(message, history):
    system_message = "You are a helpful assistant"
    history_langchain_format = [SystemMessage(content=system_message)]

    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))

    # file_context = read_markdown_files(files)

    # if file_context:
    #     message = (
    #         f"{message}\n\n"
    #         f"The user uploaded the following markdown file content:\n"
    #         f"{file_context}"
    #     )

    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = get_bedrock_client().invoke(history_langchain_format)
    return gpt_response.content


# markdown_input = gr.File(
#     label="Upload Markdown Files",
#     file_types=[".md"],
#     file_count="multiple"
# )

view = gr.ChatInterface(
    chat,
    # fn=chat,
    # additional_inputs=[markdown_input],
    api_name="gradio-chatbot",
)

if __name__ == '__main__':
    if os.getenv("DEPLOYMENT_ENVIRONMENT", "development") != "production":
        # view.launch(auth=(user, password))
        view.launch()
    else:
        view.launch(
            auth=(user, password),
            server_name="0.0.0.0",
            server_port=7860,
            root_path="/chatbot/",
        )


# load_dotenv(override=True)

# import boto3

# client = boto3.client("bedrock-runtime", region_name="ca-central-1") 

# response = client.converse(
#     modelId="us.anthropic.claude-opus-4-8",
#     messages=[
#         {
#             "role": "user",
#             "content": [{"text": "Tell me a joke."}]
#         }
#     ]
# ) 

# print(response["output"]["message"]["content"][0]["text"])