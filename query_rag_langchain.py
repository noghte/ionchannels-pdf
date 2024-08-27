import os
import re
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import Qdrant
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
MODEL_NAME = "gpt-4o"

PROMPT_TEMPLATE = """
You are a helpful assistant specialized in reading biology papers. 
Provide an evidence-based answer to the user question. 
Your response must be in 3 lines. 
1:'answer': the answer to the user's question in this format: If evidence found: `Evidence Found`, else: `Evidence Not Found`.
2:'confidence': a value between 0 and 1, where 1 indicates the highest confidence.
3:'evidence': evidence supporting the answer from the context. (your rationale)
If you don't know the answer, just say `Evidence Not Found` as the answer. Don't try to make up an answer if you cannot find the evidence in the provided context.
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

def query(query_text, db, model, pubmed_ids=None):
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
    if pubmed_ids:
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
    scores =  "\n".join(["pubmedid: " + doc.metadata.get("pubmed_id", None) + ", score: " + str(score) + "\n" for doc, score in results])

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Generate response text based on the prompt
    response_text = model.predict(prompt)
    
    # Get sources of the matching documents
    sources = "\n\n".join([f"pubmed: {doc.metadata.get('pubmed_id', None)}, text: {doc.page_content}" for doc, _score in results])

    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\n\n****Metadata****\nScores:\n{scores}\nRelevant Chunks:\n{sources}"
    return formatted_response, response_text

def extract_pubmed_ids(pubmed_ids_str):
    """
    Extracts valid PubMed IDs from a string.
    Args:
        pubmed_ids_str (str): The string containing PubMed IDs.
    Returns:
        List[str]: A list of valid PubMed IDs, or an empty list if the format is not acceptable.
    """
    # Define a regular expression pattern to match valid PubMed IDs
    pattern = r'PubMed:|PMID:\s*(\d+)'

    # Find all matches
    matches = re.findall(pattern, pubmed_ids_str)

    # Filter out any non-numeric or invalid entries
    pubmed_ids = [match for match in matches if match.isdigit()]

    return pubmed_ids if pubmed_ids else []

if __name__ == "__main__":
    # Initialize the embedding function
    embedding_function = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    # Initialize the database
    qdrant_client = QdrantClient(path="./qdrant")
    collection_info = qdrant_client.get_collection("pubmed-large")

    # Print the collection schema
    print("Collection Schema:")
    print(collection_info.dict())
    db = Qdrant(client=qdrant_client, collection_name="pubmed-large", embeddings=embedding_function)
    
    # Initialize the OpenAI chat model
    model = ChatOpenAI(model_name=MODEL_NAME, api_key=OPENAI_API_KEY)
    
    # Load the CSV file
    df = pd.read_csv('human_IC_annotation_sample.csv', dtype=object, encoding='utf-8')
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    for index, row in df.iterrows():
        uniprot_id = row["Uniprot"].strip()
        # if uniprot_id != 'Q9Y5S1':
        #     continue
        pubmed_ids_str = row["PubMed"].strip()
        ion = row["Ion"].strip()
        # family = row["Family"].strip()
        ionchannel_name = row["IonChannelName"].strip()
        ionchannel_symbol = row["IonChannelSymbol"].strip()
        gate_mechanism = row["GateMechanism"].strip()
        
        pubmed_ids = extract_pubmed_ids(pubmed_ids_str)
        
        # Process the AlternativeNames column as a list
        if pd.isna(row["AlternativeNames"]) or row["AlternativeNames"].strip() == '':
            alternative_names = []
        else:
            alternative_names = [name.strip() for name in row["AlternativeNames"].split(',')]
                
        print(f"Processing Uniprot Id {uniprot_id} in '{pubmed_ids_str}'...")

        query1 = f"Is there any evidence that `{ion}` is the ion selectivity of the {ionchannel_symbol} ion channel?"
        if len(alternative_names) > 0:
            query1 += f" Alternative names: {', '.join(alternative_names)}"

        formatted_response1, response_text1 = query(query1, db, model, pubmed_ids=pubmed_ids)
        
        query2 = f"Does this article provide evidence for `{gate_mechanism}` as the gating mechanism for the `{ionchannel_name} (symbol(s): {ionchannel_symbol})` ion channel?"
        if len(alternative_names) > 0:
            query2 += f" Alternative names: {', '.join(alternative_names)}"
        formatted_response2, response_text2 = query(query2, db, model, pubmed_ids=pubmed_ids)
        
        # Write results to a text file
        file_name = f'results-all-model_{MODEL_NAME}_{timestamp}.txt'
        with open(file_name, 'a') as f:
            f.write(f"UniProt: {uniprot_id}\n")
            f.write(f"Original PubMed: {pubmed_ids_str}\n")
            f.write(f"Query 1: {query1}\n")
            f.write(f"Results 1: {formatted_response1}\n")
            f.write(f"Query 2: {query2}\n")
            f.write(f"Results 2: {formatted_response2}\n")
            f.write("\n")
            f.write("---------------------------------------------\n")
