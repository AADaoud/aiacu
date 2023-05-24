import streamlit as st
import time
from streamlit_chat import message
import faiss
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import openai
import json

# Initialize the session state variables at the start of your script
if "generated" not in st.session_state:
    st.session_state["generated"] = []
    
if "past" not in st.session_state:
    st.session_state["past"] = []


# Use Streamlit's secret management for secure handling of API keys
api_key = st.secrets["OPENAI_API_KEY"]

# Load the LangChain.
index = faiss.read_index("docs.index")

try:
    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
        store.index = index
except FileNotFoundError:
    st.write("The file faiss_store.pkl was not found.")
except pickle.UnpicklingError:
    st.write("Could not unpickle the file faiss_store.pkl.")
except Exception as e:
    st.write(f"An unexpected error occurred: {str(e)}")


store.index = index
openai_instance = OpenAI(client=openai, openai_api_key=api_key, temperature=0)
vector_db_chain = RetrievalQAWithSourcesChain.from_chain_type(openai_instance, chain_type="map_reduce", retriever=store.as_retriever())

def handle_input(user_input):
    try:
        # First, ask GPT-3.5-turbo for a retrieval query.
        query_generation_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 'You are a helpful assistant with access to a medical knowledge database. Your task is to analyze the provided user query and determine if database retrieval is necessary. If retrieval is necessary, create a retrieval query in the form of a JSON dictionary.'},
            {"role": "user", "content": f"""
            Given the following query:

            {user_input}

            Evaluate if a medical database retrieval is necessary. If it is, formulate a JSON-formatted dictionary containing the retrieval query to the medical knowledge database. The structure should include the following keys and must include any information you deem relevant to present a comprehensive and high quality answer which is relevant to the question:

            1. "needs_retrieval": A boolean indicating whether database retrieval is necessary.
            2. "query": If retrieval is necessary, identify the main keywords or phrases that summarize the user's medical question. If retrieval is not necessary, this should be an empty string.

            Here are examples of the desired structures:

            For a query that requires database retrieval, such as "What is diabetes?":
                {{
                    "needs_retrieval": true,
                    "query": "provide a comprehensive overview of diabetes, including its types, their causes, symptoms, and treatment methods."
                }}

            If the query does not contain any relevant medical terminology and does not require the retrieval of information from the medical knowledge database (such as "How is the weather today?", "How are you?", or any other non-medical queries that can be asked in non-academic and non-medical settings), formulate a JSON-formatted dictionary exactly as follows:
                {{
                    "needs_retrieval": false,
                    "query": ""
                }}

            Please respond with a JSON structure ONLY.
            """}],
        temperature=0.2,
        max_tokens=500
    )

        # Extract the query from GPT-3.5-turbo's response.
        query_generation_output = query_generation_response['choices'][0]['message']['content']
        print(query_generation_output, type(query_generation_output))
        query_json = json.loads(query_generation_output)
        query_json = json.loads(query_generation_output.strip())


        if query_json['needs_retrieval']:
            retrieval_query = query_json['query']

            # Use the generated query to get a result from the document retrieval system.
            retrieval_result = vector_db_chain({"question": retrieval_query})
            print(retrieval_result)

            # Ask GPT-3.5-turbo to formulate a response based on the retrieved documents.
            response_generation_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant skilled in interpreting medical information. A user asked this question: '{user_input}'. Here are the documents that were retrieved based on their query from a medical database: {json.dumps(retrieval_result)}. Please formulate a comprehensive and high-quality answer in natural language that incorporates all the information provided. The answer should be suitable for a medical student and presented in a logical and coherend manner."},
                    {"role": "user", "content": json.dumps(retrieval_result)}],
                temperature=0.5,
                max_tokens=2000
            )

            # Extract the response from GPT-3.5-turbo's response.
            response_generation_output = response_generation_response['choices'][0]['message']['content']

        else:
            # No retrieval necessary; generate a standard response
            response_generation_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a happy and helpful AI medical assistant who provides medical students with answers to their questions. As a helpful medical assistant try to be helpful, polite, honest, sophisticated, emotionally aware, and knowledgeable."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.7
            )
            response_generation_output = response_generation_response['choices'][0]['message']['content']

        return response_generation_output

    except json.JSONDecodeError as e:
        return f"Could not parse query from GPT-3.5-turbo's response: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


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
