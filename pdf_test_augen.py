import os
import augen   
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent
from qdrant_client import QdrantClient

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "gpt-4o"

llm_config = {
    "config_list": [{"model": MODEL_NAME, "api_key": os.environ["OPENAI_API_KEY"]}],
}

# Print the generation steps
autogen.ChatCompletion.start_logging()

# 1. create a RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "request_timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

# 2. create a QdrantRetrieveUserProxyAgent instance named "qdrantagent"
# By default, the human_input_mode is "ALWAYS", i.e. the agent will ask for human input at every step.
# `docs_path` is the path to the docs directory.
# `task` indicates the kind of task we're working on.
# `chunk_token_size` is the chunk token size for the retrieve chat.
# We use an in-memory QdrantClient instance here. Not recommended for production.

rag_proxy_agent = QdrantRetrieveUserProxyAgent(
    name="qdrantagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": "./path/to/docs",
        "chunk_token_size": 2000,
        "model": MODEL_NAME,
        "client": QdrantClient(":memory:"),
        "embedding_model": "BAAI/bge-small-en-v1.5",
    },
)
