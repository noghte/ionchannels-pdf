import os
import pandas as pd
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
    system_message="You are a helpful assistant. When receiving a message from the user, it is your responsibility to provide an evidence-based answer. Your response must be in 3 lines. 1: 'answer', 2:'confidence', 3:'evidence'. The 'answer' line contain the answer to the user's question as one word of 'Yes' or 'No'. The 'confidence' line must contain a value between 0 and 1, where 1 indicates the highest confidence. The 'evidence' line must contain a list of evidence supporting the answer from the provided documents. Each piece of evidence has two lines: 1. your rationale in answering the user question. 2. the orginal texts you decided based on that. If you don't know the answer, just say `No` as the answer. Don't try to make up an answer if you cannot find the evidence in the provided context.",
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
DESCRIPTION = 'A biology expert that finds relevant information regarding ion channels in provided texts and responses the queries with `yes` or `no`'

PDF_DIR = "./pdf"
docs = []
for pdf_file in os.listdir(PDF_DIR):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        docs.append(pdf_path)

# iterate over data.csv
df = pd.read_csv('human_IC_annotation_sample.csv',dtype=object, encoding='utf-8')

for index, row in df.iterrows():
    uniprot_id = row["Uniprot"].strip()
    pubmed_id = row["PubMed"].strip()
    collection_name = "pubmed_" + pubmed_id
    print(f"Processing Uniprot Id {uniprot_id} in '{collection_name}'...")
    #
    # collection_name = "pubmed_10484328"
    # The ragproxyagent retrieves document chunks based on the embedding similarity, and sends them along with the question to the assistant.
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
    ion = row["Ion"].strip()
    family = row["Family"].strip()
    ionchannel_name = row["IonChannelName"].strip()
    ionchannel_symbol = row["IonChannelSymbol"].strip()
    gate_mechanism = row["GateMechanism"].strip()
    # query1 = f"Is there any evidence that `{ion}` is the ion selectivity of the `{family}` ion channel?"
    query1 = f"Is there any evidence that `{ion}` is the ion selectivity of the `{ionchannel_name}(symbol: {ionchannel_symbol})` ion channel?"

    chat_results1 = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=query1)
    

    query2 = f"Does this article provide evidence for `{gate_mechanism}` as the gating mechanism for the `{family}` ion channel?"
    chat_results2 = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=query2)
    #print(chat_results.summary)
    with open('results.txt', 'a') as f:
        f.write(f"UniProt: {uniprot_id}\n")
        f.write(f"PubMed: {pubmed_id}\n")
        f.write(f"Query 1: {query1}\n")
        f.write(f"Results 1: {chat_results1.summary}\n")
        f.write(f"Query 2: {query2}\n")
        f.write(f"Results 2: {chat_results2.summary}\n")
        f.write("\n")
        f.write("---------------------------------------------\n")
