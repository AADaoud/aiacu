import streamlit as st
from streamlit_chat import message
import faiss
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import openai

# Use Streamlit's secret management for secure handling of API keys
api_key = st.secrets["OPENAI_API_KEY"]

# Load the LangChain.
index = faiss.read_index("docs.index")

try:
    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
except Exception as e:
    st.write(f"An error occurred while loading the pickle file: {str(e)}")

store.index = index
openai_instance = OpenAI(client=openai, openai_api_key=api_key, temperature=0)
vector_db_chain = RetrievalQAWithSourcesChain.from_chain_type(openai_instance, chain_type="stuff", retriever=store.as_retriever())

# Load keywords from set.txt file
with open("set.txt", "r") as f:
    keywords = set(line.strip() for line in f)

def handle_input(user_input, use_gpt_only):
    if use_gpt_only or not any(keyword in user_input for keyword in keywords):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful medical student assistant, please answer the query of the student as best as you can."},
                        {"role": "user", "content": user_input}],
            temperature=0.5,
            max_tokens=150
        )
        output = response['choices'][0]['message']['content']
    else:
        result = vector_db_chain({"question": user_input})
        output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    return output


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Medical Question Bot-ACU-Edition", page_icon=":robot:")
st.header("Medical Question Bot: ACU-Edition")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

user_input = get_text()

use_gpt_only = st.checkbox("Use GPT only")

if user_input:
    output = handle_input(user_input, use_gpt_only)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
