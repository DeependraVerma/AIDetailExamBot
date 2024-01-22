import os
import streamlit as st
import pickle
import time
import langchain
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv() 
custom_icon = "www.vectorstock.com%2Froyalty-free-vector%2Fschool-supplies-study-icon-vector-10760940&psig=AOvVaw270djIKZp8biIyaVU2YLbE&ust=1706023535721000&source=images&cd=vfe&ved=0CBMQjRxqFwoTCKjvn6mn8YMDFQAAAAAdAAAAABAE"
st.set_page_config(page_title="Exambot: AI tool 🤖", page_icon=custom_icon)
st.title("Exambot: Ask AI 🤖")
st.sidebar.title("News Article URLs")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_genai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)


    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_genai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n") 
                for source in sources_list:
                    st.write(source)