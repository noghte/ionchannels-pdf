import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Constants
PDF_DIR = "./pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

# Iterate over all .pdf files in the PDF_DIR
for pdf_file in os.listdir(PDF_DIR):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        collection_name = f"pubmed_{os.path.splitext(pdf_file)[0]}"

        # Load the PDF document
        loader = PDFMinerLoader(pdf_path)
        document = loader.load()

        # Split the document into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n\n", 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(document)

        # Print information for debugging purposes
        print(f"Processing '{pdf_file}'...")
        print("Number of paragraphs:", document[0].page_content.count("\n\n"))
        print("Number of chunks:", len(docs))

        # Create or recreate the Chroma collection for the current PDF
        chroma_path = f"./chroma_db/"

        # Create or recreate the Chroma collection for the current PDF
        chroma = Chroma.from_documents(
            docs, 
            embeddings,
            persist_directory=chroma_path,
            collection_name=collection_name
        )
        chroma.persist()  # Persist the collection to disk
        del chroma # to release the memory and allow the next collection to be created
        print(f"Collection '{collection_name}' created successfully.\n")
        
print("All PDF files processed.")
