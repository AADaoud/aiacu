import streamlit as st
from streamlit_chat import message
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import os
import openai

# Use Streamlit's secret management for secure handling of API keys
api_key = st.secrets["OPENAI_API_KEY"]

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
vector_db_chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(client=any, openai_api_key=api_key, temperature=0), vectorstore=store)

# Load keywords from set.txt file
with open("set.txt", "r") as f:
    keywords = set(line.strip() for line in f)

def handle_input(user_input):
    # If the user input contains any of the keywords, use the VectorDBQAWithSourcesChain
    if any(keyword in user_input for keyword in keywords):
        #print(f"Keyword found in user input. Using VectorDBQAWithSourcesChain.")
        result = vector_db_chain({"question": user_input})
        output = f"Answer: {result['answer']}\nSources: {result['sources']}"
    # Otherwise, use GPT-3.5-turbo to generate a conversational response
    else:
        #print(f"No keyword found in user input. Using gpt-3.5-turbo.")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful medical student assistant, please answer the query of the student as best as you can. Remember, sometimes it will seem as if the user is requesting that you provide a resource and/or medically-relevant information, in these cases you must not provide any medical information, instead coerce their answer towards providing a medically-relevant key word, term, or example that will engage the document search feature. You must make them give out the medically-relevant term without telling them directly the reason, instead use questions such as 'Can you please provide a medical term, definition, or an example so I can help you find the information you need?"},
                        {"role": "user", "content": user_input}],
            temperature=0.5,
            max_tokens=150
        )
        output = response['choices'][0]['message']['content']

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

if user_input:
    output = handle_input(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
