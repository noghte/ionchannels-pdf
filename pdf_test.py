# from PyPDF2 import PdfReader
import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
import autogen
load_dotenv()

PDF_PATH = "./pdf/10484328.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# reader = PdfReader(PDF_PATH)

# pages = reader.pages
# documents = []

# for page in pages:
#     documents.append(page.extract_text())
# print("Length:", len(documents))
# print("Document:",documents[0])

loader = PDFMinerLoader(PDF_PATH)
document = loader.load()
# find number of \n\n in document[0]
print("Number of paragraphs", document[0].page_content.count("\n\n"))
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, )
docs = text_splitter.split_documents(document)
print("Length:", len(docs))
# print("Document 1:", docs[0])
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

qdrant = Qdrant.from_documents(
    docs, 
    embeddings,
    path="./qdrant",
    collection_name="pubmed",
    force_recreate=True
    )

    # ion selectivity: water, potassium, magni..abs
    # gating mechansim: voltage, ligand, mechanosensitive, temprature
# query = "Does this article mention that pottasium is the ion selectivity of the K_V ion channel?"
# answer = "yes/no"
# answer = "The selectivity of the K (or K_V) channel is pottasium."

query = "Does this article provide evidence for voltage-gated mechanism for the K_V ion channel?"
answer = "yes/no" 

docs = qdrant.similarity_search(query)

print("Answer:", docs[0].page_content)