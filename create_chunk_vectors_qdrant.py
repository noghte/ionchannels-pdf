import os
import re
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
PDF_DIR = "./pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large" #embedding size small: 1536, large: 3072
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def create_or_get_collection(qdrant_client, collection_name, embeddings):
    try:
        # Check if the collection already exists
        response = qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' found.")
    except Exception as e:
        print(f"Resolving this error: {e}")
        # If the collection does not exist, create it
        print(f"Collection '{collection_name}' not found. Creating a new collection.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 3072,# embeddings.dimensions,  # specify the embedding size
                "distance": "Cosine"  # choose the distance metric, e.g., Cosine, Euclidean, etc.
            }
        )
        print(f"Collection '{collection_name}' created.")
    
    # Initialize the Qdrant collection
    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    return qdrant

def remove_unwanted_lines(text):
    # Remove lines that start with "Downloaded from ..."
    filtered_lines = []
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line.lower().startswith("downloaded from"):
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

def is_reference_chunk(text):
    # Define a pattern that matches common reference formats, like numbers followed by author names and dates
    reference_patterns = [
        r"^\[\d+\]",  # Matches reference numbers in brackets, e.g., "[136]"
        r"^\d{1,2}\.",  # Matches numbers at the start, e.g., "27. Nimigean, C. M."
        r"\b\(\d{4}\)\b",  # Matches year in parentheses, e.g., "(2011)"
        r"\bdoi:|https?://",  # Matches DOI and URLs
        # r"\bJ\.|Nature|Cell|Biophys\. J\.|Elife\b",  # Matches common journal abbreviations and names
        r"\bet\ al\.|pp\.\s+\d+-\d+\b",  # Matches "et al." or page ranges
        r"\b[A-Z][a-z]+,\s+[A-Z]\.\s+[A-Z][a-z]+\b",  # Matches author names, e.g., "J. Rettig, S.H. Heinemann". It captures a capitalized last name, followed by a comma, initials, and another capitalized last name.        
    ]
    
    # Combine all patterns into a single pattern
    combined_pattern = re.compile("|".join(reference_patterns), re.IGNORECASE)
    
    # Check if the chunk matches any of the reference patterns
    return bool(combined_pattern.search(text))

def update_collection(pdf_file, qdrant):
    # Load the PDF document
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    loader = PDFMinerLoader(pdf_path)
    document = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n", 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(document)

    filtered_chunks = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        text = remove_unwanted_lines(text)
        if not is_reference_chunk(text):
            filtered_chunks.append(chunk)
    
    # Assign PubMed ID to filtered chunks
    for chunk in filtered_chunks:
        chunk.metadata["pubmed_id"] = pdf_path.split("/")[-1].split(".")[0]

    # Add filtered chunks to the existing Qdrant collection
    qdrant.add_documents(filtered_chunks)

    print(f"'{pdf_file}' processed and vectors (excluding references) added to the collection.")


def create_main_collection(collection_name):
    # Initialize the Qdrant client
    qdrant_client = QdrantClient(path="./qdrant")

    # Create or get the Qdrant collection
    qdrant = create_or_get_collection(qdrant_client, collection_name, embeddings_model)
    # process = False
    ignore_files = ["11852086.pdf", "25150048.pdf"]

    # Process each PDF and update the Qdrant collection
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf") and pdf_file not in ignore_files:
            print(f"Processing '{pdf_file}'...")
            update_collection(pdf_file, qdrant)
    
    print("All PDF files processed and vectors added to the collection.")

if __name__ == "__main__":
    create_main_collection(collection_name="pubmed-large")
