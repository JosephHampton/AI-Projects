from pathlib import Path
from typing import List, Tuple

from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field
from langchain.chains import ConversationalRetrievalChain

# Constants
local_path = "./models/gpt4all-lora-unfiltered-quantized.new.bin"
model_path = "./models/gpt4all-lora-unfiltered-quantized.new.bin"
text_path = "./docs/conversation.txt"
index_path = "./conversation_index"

# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=local_path)

def load_documents() -> List:
    loader = TextLoader(text_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
llm = GPT4All(model=local_path, n_ctx=2048,n_threads=8, verbose=True)

embeddings = initialize_embeddings()
#sources = load_documents()
#chunks = split_chunks(sources)
#vectorstore = generate_index(chunks, embeddings)
#vectorstore.save_local("conversation_index")

index = FAISS.load_local(index_path, embeddings)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=200)

# Chatbot loophow are you today

chat_history = []
print("Welcome to Joe's chatbot! Type 'exit' to stop.")
while True:
    query = input("Please enter your question: ")
    
    if query.lower() == 'exit':
        break
    result = qa({"question": query, "chat_history": chat_history})
    print(result['answer'])