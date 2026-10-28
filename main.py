import os
import boto3
import gradio as gr
from botocore.exceptions import ClientError

# ---------- Configuration ----------

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-5-5")
AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def to_bedrock_messages(history, user_message):
    """Convert Gradio 'messages'-style history into Bedrock Converse format."""
    messages = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        # Skip non-text content (e.g. file components) and unknown roles
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        # Converse requires alternating roles; merge consecutive same-role turns
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"][0]["text"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": [{"text": content}]})

    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"][0]["text"] += "\n\n" + user_message
    else:
        messages.append({"role": "user", "content": [{"text": user_message}]})

    # Conversation must start with a user message
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages
 

def chat(user_message, history, system_prompt = "You are a helpful assistant.", max_tokens = 128000, temperature = 1.0):
    messages = to_bedrock_messages(history, user_message)
    request = {
        "modelId": MODEL_ID,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
        },
    }
    if system_prompt and system_prompt.strip():
        request["system"] = [{"text": system_prompt.strip()}]

    try:
        response = bedrock.converse_stream(**request)
        partial = ""
        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    partial += delta["text"]
                    yield partial
            elif "messageStop" in event:
                reason = event["messageStop"].get("stopReason")
                if reason == "max_tokens":
                    partial += "\n\n*[Response truncated: max tokens reached]*"
                    yield partial
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        hint = ""
        if code == "ValidationException" and "model" in msg.lower():
            hint = "\n\nCheck that BEDROCK_MODEL_ID is correct; you may need an inference profile ID."
        elif code == "AccessDeniedException":
            hint = "\n\nEnable model access in the Bedrock console and check your IAM permissions."
        yield f"**{code}**: {msg}{hint}"
    except Exception as e:
        yield f"Unexpected error: {e}"


view = gr.ChatInterface(
        fn=chat
    )

if __name__ == "__main__":
    view.launch()