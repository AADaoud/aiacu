from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import re

# Use Streamlit's secret management for secure handling of API keys
api_key = st.secrets["OPENAI_API_KEY"]

ps = list(Path(".Lectures/").glob("**/*.md"))

data = []
sources = []
topics = []
for p in ps:
    with open(p, encoding="utf-8", errors='replace') as f:
        content = f.read()
        data.append(content)
        topic_match = re.search(r"# Key Topics\n((?:.+\n)*)", content)
        if topic_match:
            topics.append(topic_match.group(1).strip())
        else:
            topics.append(None) 
    sources.append(str(p)) # convert pathlib.Path to string here

text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    if topics[i] is not None:
        metadatas.extend([{"source": sources[i], "header": topics[i]}] * len(splits))
    else:
        metadatas.extend([{"source": sources[i]}] * len(splits))

store = FAISS.from_texts(docs, OpenAIEmbeddings(client=any, openai_api_key=api_key), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
