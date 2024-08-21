import os
from autogen.agentchat import AssistantAgent
import chromadb.utils.embedding_functions as embedding_functions
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

# embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
# query_vector = embeddings.embed_query("plasmid constructions")

MODEL_NAME = "gpt-4o"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.list_collections()
print("Available collections:", [c.name for c in collection])

collection_name = "pubmed_10484328"
# if not client.collection_exists(collection_name=collection_name):
#     print(f"Collection '{collection_name}' does not exist. Please create the collection first.")
#     exit()


llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": [{"model": MODEL_NAME, "api_key": os.environ["OPENAI_API_KEY"]}],
}
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant. You have one job. Job 1. When receiving a message from the user, it is your responsibility to provide an evidence-based answer. Your response must be in JSON format. The JSON response must contain the following keys: 'answer', 'confidence', 'evidence' (cite the source). The 'answer' key must contain the answer to the user's question as 'yes' or 'no'. The 'confidence' key must contain a value between 0 and 1, where 1 indicates the highest confidence. The 'evidence' key must contain a list of evidence supporting the answer. Each piece of evidence must be a dictionary with the following keys: 'text', 'source'. If you don't know the answer, just say `no` as the answer. Don't try to make up an answer.",
    llm_config=llm_config,
)

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
PDF_DIR = "./pdf"
docs = []
for pdf_file in os.listdir(PDF_DIR):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        docs.append(pdf_path)

# The ragproxyagent retrieves document chunks based on the embedding similarity, and sends them along with the question to the assistant.
DESCRIPTION = 'A biology expert that finds relevant information regarding ion channels in provided texts and responses the queries with `yes` or `no`'
ragproxyagent = RetrieveUserProxyAgent(
    name="pdfagent",
    human_input_mode="NEVER",
    is_termination_msg=None,
    max_consecutive_auto_reply=5,
    retrieve_config={ # https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/retrieve_user_proxy_agent/
        "task": "qa", 
        "vector_db": "chroma",
        "description": DESCRIPTION,
        "client": client, 
        "docs_path": docs,
        "collection_name": collection_name,
        "get_or_create": True,
        "overwrite": False,
        "clean_up_tokenization_spaces": True,
        # "embedding_function": openai_ef,
        # "embedding_model": OPENAI_EMBEDDING_MODEL,
        "model": MODEL_NAME,
        }
)
ragproxyagent.description = DESCRIPTION


assistant.reset() # to forget previous conversations
query = "Is there any evidence that pottasium is the ion selectivity of the K_V ion channel?"
chat_results = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=query)
print(chat_results.summary)
