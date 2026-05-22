

import os
import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#Ingestion
def load_pdf(pdf_path):
    documents = []
    pdf = fitz.open(pdf_path)
    for page_num, page in enumerate(pdf):
        text = page.get_text()
        if len(text.strip()) < 100:
            continue
        doc = Document(
            page_content=text,
            metadata={
                "page": page_num + 1,
                "source": pdf_path
            }
        )
        documents.append(doc)
    return documents

#chunking 

def chunking(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    return chunks

#embeddings
def embeddings_get():


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

#saving embeddings into vector database
def vectore_store(chunks):
    embeddings=embeddings_get()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore
    
    


###

def main():
    print("Hello from rag-qa!")
    docs = load_pdf("data/apple_2023.pdf")
    print("Total pages loaded:", len(docs))
    chunks =chunking(docs)
    print("Total chunks:", len(chunks))
   #print(chunks[0].page_content[:500])
    vectorstore =vectore_store(chunks)
    print("Vector DB created")
    results = vectorstore.similarity_search(
    "What was Apple's total revenue?",
        k=3
    )
    for r in results:
        print("\nPAGE:", r.metadata["page"])
        print(r.page_content[:500])

    


if __name__ == "__main__":
    main()