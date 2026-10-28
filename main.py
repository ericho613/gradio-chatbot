import os
import boto3
import gradio as gr
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError

# ---------- Configuration ----------

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-opus-5-5")
AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")

bedrock_config = Config(
    connect_timeout=10,     # fail fast if the endpoint is unreachable
    read_timeout=3600,      # allow long generations (AWS guidance for Claude)
    retries={
        "max_attempts": 2,  # don't silently retry long requests many times
        "mode": "standard",
    },
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=bedrock_config)


def extract_text(content):
    """Pull plain text out of the various content shapes Gradio may send."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # Gradio 6 content block: {"type": "text", "text": "..."}
        if content.get("type", "text") == "text" and isinstance(content.get("text"), str):
            return content["text"]
        return ""  # files, images, etc.
    if isinstance(content, (list, tuple)):
        return "\n".join(t for t in (extract_text(c) for c in content) if t)
    return ""  # file components and other non-text objects


def normalize_history(history):
    """Yield (role, text) pairs from any Gradio history format."""
    for turn in history or []:
        # Old "tuples" format: [user_msg, assistant_msg]
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            user, assistant = turn
            yield "user", extract_text(user)
            yield "assistant", extract_text(assistant)
            continue
        # "messages" format: dict or gr.ChatMessage object
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
        else:
            role, content = getattr(turn, "role", None), getattr(turn, "content", None)
        yield role, extract_text(content)


def to_bedrock_messages(history, user_message):
    """Convert Gradio chat history into Bedrock Converse format."""
    messages = []
    for role, text in normalize_history(history):
        if role not in ("user", "assistant") or not text.strip():
            continue
        # Converse requires alternating roles; merge consecutive same-role turns
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"][0]["text"] += "\n\n" + text
        else:
            messages.append({"role": role, "content": [{"text": text}]})

    user_text = extract_text(user_message)  # handles multimodal dict input too
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"][0]["text"] += "\n\n" + user_text
    else:
        messages.append({"role": "user", "content": [{"text": user_text}]})

    # Conversation must start with a user message
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def chat(user_message, history, system_prompt="You are a helpful assistant.",
         max_tokens=4096):
    messages = to_bedrock_messages(history, user_message)
    request = {
        "modelId": MODEL_ID,
        "messages": messages,
        "inferenceConfig": {"maxTokens": int(max_tokens)},
    }
    if system_prompt and system_prompt.strip():
        request["system"] = [{"text": system_prompt.strip()}]

    partial = ""
    try:
        response = bedrock.converse_stream(**request)
        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    partial += delta["text"]
                    yield partial
            elif "messageStop" in event:
                if event["messageStop"].get("stopReason") == "max_tokens":
                    partial += "\n\n*[Response truncated: max tokens reached]*"
                    yield partial
    except (ReadTimeoutError, ConnectTimeoutError) as e:
        yield partial + (
            f"\n\n**Timeout**: {e}\n\nThe model took too long to respond. "
            "Try a smaller max_tokens value, or check network/VPC access to "
            f"bedrock-runtime.{AWS_REGION}.amazonaws.com."
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        hint = ""
        if code == "ValidationException" and "model" in msg.lower():
            hint = "\n\nCheck that BEDROCK_MODEL_ID is correct; you may need an inference profile ID."
        elif code == "AccessDeniedException":
            hint = "\n\nEnable model access in the Bedrock console and check your IAM permissions."
        elif code == "ThrottlingException":
            hint = "\n\nYou're being rate limited; wait and retry, or request a quota increase."
        yield partial + f"\n\n**{code}**: {msg}{hint}"
    except Exception as e:
        yield partial + f"\n\nUnexpected error: {type(e).__name__}: {e}"


# Request the "messages" history format where supported (Gradio 4.x/5.x).
# Gradio 6 removed the `type` argument and always uses messages.
try:
    view = gr.ChatInterface(fn=chat, type="messages")
except TypeError:
    view = gr.ChatInterface(fn=chat)

if __name__ == "__main__":
    view.launch()