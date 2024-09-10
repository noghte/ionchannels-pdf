import os
import json
import re
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import Qdrant
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

IS_LOCAL_MODEL = True
INPUT_FILE = "human_IC_annotation_sample.csv" #"human_IC_annotation.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
MODEL_NAME = "gpt-4o"

PROMPT_TEMPLATE = """
You are a helpful assistant specialized in reading biology papers. 
Provide an evidence-based answer to the user question. 
Your response must be in JSON format with the following keys:
-'answer': the answer to the user's question in this format: If evidence found: `Found`, else: `Not Found`. If you don't have access to the answer in the context, just say `Not Found` as the answer. Don't try to make up an answer if you cannot find the evidence in the provided context.
-'confidence': a value between 0 and 1, where 1 indicates the highest confidence.
-'evidence': evidence supporting the answer from the context. (your rationale)

NOTES:
- The JSON response should be in one line, valid, and have no syntax errors.
- The ion channel name should be the exact name or one of the alternative names provided in the context.
- The response should be only in JSON, not any extra text before or after the JSON.
- Make sure that the context is about the specified ion channel (or one of its alternative names), otherwise return `Not Found` as the answer.

Answer the question based only on the following context. If the answer is not in the context, return `Not Found` as the answer.:
{context}
 - -
Answer the question based on the above context: {question}
"""

class IonChannelResponse(BaseModel):
    answer: str = Field(description="the answer to the user's question in this format: If evidence found: `Found`, else: `Not Found`.")
    confidence: float = Field(description="a vadb.similarity_search_with_relevance_scores(query_text, k=3, filter=filters)lue between 0 and 1, where 1 indicates the highest confidence.")
    evidence: str = Field(description="evidence supporting the answer from the context.")

def query(query_text, db, model, ion_channel_names, pubmed_ids=None):
    """
    Query a Retrieval-Augmented Generation (RAG) system using a vector database and OpenAI.
    Args:
      - query_text (str): The text to query the RAG system with.
      - db (Qdrant): The Qdrant instance used to perform similarity searches.
      - model (ChatOpenAI): The OpenAI chat model instance used to generate responses.
      - pubmed_ids (list[str], optional): List of PubMed IDs to filter the vector search.
    Returns:
      - formatted_response (str): Formatted response including the generated text and sources.
      - response_text (str): The generated response text.
    """
    # Construct the filter for PubMed IDs
    filters = None
    if pubmed_ids and len(pubmed_ids) > 0:
        filters = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.pubmed_id",
                    match=models.MatchValue(value=pubmed_id)
                ) for pubmed_id in pubmed_ids
            ]
        )

    # Retrieving the context from the DB using similarity search with filters
    filtered_results = db.similarity_search_with_relevance_scores(query_text, k=3, filter=filters)

    # Retrieve additional context from the DB without filters
    unfiltered_results = db.similarity_search_with_relevance_scores(query_text, k=3, filter=None)

    combined_results = {doc.id: (doc, score) for doc, score in filtered_results}
    for doc, score in unfiltered_results:
        if doc.metadata["_id"] not in combined_results:
            combined_results[doc.metadata["_id"]] = (doc, score)

    # Sort combined results by score
    results = sorted(combined_results.values(), key=lambda x: x[1], reverse=True)
    
    # Check if there are any matching results
    if len(results) == 0:
        print(f"Unable to find matching results for: {query_text}")
        return None, None

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
    # scores =  "\n".join(["pubmedid: " + doc.metadata.get("pubmed_id", None) + ", score: " + str(score) + "\n" for doc, score in results])

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    
    parser = JsonOutputParser(pydantic_object=IonChannelResponse)
    chain = prompt_template | model | parser
    # Generate response text based on the prompt
    # response_text = model.predict(prompt)
    response_json = chain.invoke({"context": context_text, "question": query_text})
    
    # Get sources of the matching documents
    # sources = "\n\n".join([f"pubmed: {doc.metadata.get('pubmed_id', None)}, text: {doc.page_content}" for doc, _score in results])
    sources_list = [
        {
            "pubmed_id": doc.metadata.get('pubmed_id', None),
            "text": doc.page_content,
            "score": round(score, 4)
        }
        for doc, score in results
    ]

    response_json['sources'] = sources_list
    # Format and return response including generated text and sources
    # formatted_response = f"Response: {response_text}\n\n****Metadata****\nScores:\n{scores}\nRelevant Chunks:\n{sources}"
    return response_json

def extract_pubmed_ids(pubmed_ids_str):
    """
    Extracts valid PubMed IDs from a string.
    Args:
        pubmed_ids_str (str): The string containing PubMed IDs.
    Returns:
        List[str]: A list of valid PubMed IDs, or an empty list if the format is not acceptable.
    """
    if not pubmed_ids_str:
        return []

    # Define a regular expression pattern to match valid PubMed IDs
    pattern = r'PubMed:|PMID:\s*(\d+)'

    # Find all matches
    matches = re.findall(pattern, pubmed_ids_str)

    # Filter out any non-numeric or invalid entries
    pubmed_ids = [match for match in matches if match.isdigit()]

    return pubmed_ids if pubmed_ids else []

def save_results_to_json(file_name, uniprot_id, res_json1, res_json2):
    # Check if the file exists
    if os.path.exists(file_name):
        # Load the existing data
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        # Create an empty dictionary if the file doesn't exist
        data = {}

    # Add or update the data under the uniprot_id
    data[uniprot_id] = {
        "query1": res_json1,
        "query2": res_json2
    }

    # Save the updated data back to the file
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Results of the uniprot {uniprot_id} saved in the json file.")
if __name__ == "__main__":
    # Initialize the embedding function
    embedding_function = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    # Initialize the database
    qdrant_client = QdrantClient(path="./qdrant")
    collection_info = qdrant_client.get_collection("pubmed-large")

    # Print the collection schema
    print("Collection Schema:")
    print(collection_info.model_dump_json())
    db = Qdrant(client=qdrant_client, collection_name="pubmed-large", embeddings=embedding_function)
    
    # Initialize the OpenAI chat model
    if IS_LOCAL_MODEL:
        model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0)
    else:
        model = ChatOpenAI(model_name=MODEL_NAME, api_key=OPENAI_API_KEY)
    
    # Load the CSV file
    df = pd.read_csv(INPUT_FILE, dtype=object, encoding='utf-8')
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    process = False
    for index, row in df.iterrows():
        uniprot_id = row["Uniprot"].strip()
        # if uniprot_id == 'Q03721':
        #     process = True
        # if not process:
        #     continue
        
        pubmed_ids_str = row["PubMed"]
        if pd.notna(pubmed_ids_str) and isinstance(pubmed_ids_str, str):
            pubmed_ids_str = pubmed_ids_str.strip()
        else:
            pubmed_ids_str = ""  # Handle the NaN or non-string case
        ion = row["Ion"].strip()
        # family = row["Family"].strip()
        ionchannel_name = row["IonChannelName"].strip()
        ionchannel_symbol = row["IonChannelSymbol"].split("(")[0].strip()
        gate_mechanism = row["GateMechanism"].strip()
        
        pubmed_ids = extract_pubmed_ids(pubmed_ids_str)
        
        # Process the AlternativeNames column as a list
        if pd.isna(row["AlternativeNames"]) or row["AlternativeNames"].strip() == '':
            alternative_names = []
        else:
            alternative_names = [name.strip() for name in row["AlternativeNames"].split(',')]
                
        print(f"Processing Uniprot Id {uniprot_id} in '{pubmed_ids_str}'...")

        ion_channel_str = f"{ionchannel_symbol}"
        if len(alternative_names) > 0:
            ion_channel_str += f" Alternative names: {', '.join(alternative_names)}"

        query1 = f"Is there any evidence that `{ion}` is the ion selectivity of the `{ion_channel_str}` ion channel?"
        res_json1 = query(query1, db, model, ion_channel_str, pubmed_ids=pubmed_ids)
        query1_dict = {"text": query1}
        final_json1 = {**query1_dict, **res_json1}

        query2 = f"Does this article provide evidence for `{gate_mechanism}` as the gating mechanism for the `{ion_channel_str}` ion channel?"
        res_json2 = query(query2, db, model, ion_channel_str, pubmed_ids=pubmed_ids)
        query2_dict = {"text": query2}
        final_json2 = {**query2_dict, **res_json2}
    
        # Write results to a text file
        # file_name = f'results-all-model_{MODEL_NAME}_{timestamp}.json'
        if IS_LOCAL_MODEL:
            file_name = f'results-all-local_model.json'
        else:
            file_name = f'results-all-model_{MODEL_NAME}.json'
        save_results_to_json(file_name, uniprot_id, final_json1, final_json2)        

        # with open(file_name, 'a') as f:
        #     f.write(f"UniProt: {uniprot_id}\n")
        #     f.write(f"Original PubMed: {pubmed_ids_str}\n")
        #     f.write(f"Query 1: {query1}\n")
        #     f.write(f"Query 2: {query2}\n")
        #     f.write("\n")
        #     f.write("---------------------------------------------\n")
