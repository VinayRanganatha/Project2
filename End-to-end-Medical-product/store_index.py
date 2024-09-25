from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import chroma
#import chroma
from langchain import embeddings
from dotenv import load_dotenv
import os

load_dotenv()


# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone


#Creating Embeddings for Each of The Text Chunks & storing

