import os
# from autogen import AssistantAgent
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# openai_ef = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

# embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
# query_vector = embeddings.embed_query("plasmid constructions")
# client = QdrantClient(path="./qdrant")
print("Start")
MODEL_NAME = "gpt-4o"

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": [{"model": MODEL_NAME, "api_key": os.environ["OPENAI_API_KEY"]}],
}

# # 1. create an AssistantAgent instance named "assistant"
# assistant = AssistantAgent(
#     name="assistant",
#     system_message="You are a helpful assistant.",
#     llm_config=llm_config,
# )

# Optionally create embedding function object
# sentence_transformer_ef = SentenceTransformer("all-distilroberta-v1").encode
# client = QdrantClient(":memory:")

# 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
# Refer to https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/retrieve_user_proxy_agent
# and https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/vectordb/qdrant
# for more information on the RetrieveUserProxyAgent and QdrantVectorDB
ragproxyagent = QdrantRetrieveUserProxyAgent(
    name="qdrantagent",
    human_input_mode="NEVER",
    is_termination_msg=None,
    max_consecutive_auto_reply=5,
    retrieve_config={
        "task": "default", 
        "client": QdrantClient(path="./qdrant"), 
        "collection_name": "pubmed_10484328",
        "embedding_model": OPENAI_EMBEDDING_MODEL
        }
)
query = "What do you know about plasmid constructions?"
result = ragproxyagent.initiate_chat(assistant, problem=query)
print(result)