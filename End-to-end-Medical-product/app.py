from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from src.helper import load_pdf,text_split
#import chroma
import warnings
from langchain.prompts import PromptTemplate
from langchain import embeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import streamlit as st
import os


load_dotenv()

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone

persist_directory = 'db'
vectordb = Chroma.from_documents(documents=text_chunks, embedding = embeddings, persist_directory = persist_directory)

vectordb.persist()
#vectordb = none
#vectordb = Chroma(persist_directory = persist_directory, embedding_function = embeddings)


#Loading the index


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever = vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('/n/nsources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

warnings.filterwarnings("ignore")


st.title("test")

with st.form("user_inputs"):

    input=st.text_input("input",max_chars=20)

    button=st.form_submit_button("submit")

    if button and input:
        with st.spinner("loading..."):
            response=qa(
                {
                    "input":input
                }
            )
    query = input
    llm_response = qa(query)
    process_llm_response(llm_response)

    st.write(response)