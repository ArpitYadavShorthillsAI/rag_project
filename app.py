import streamlit as st
import csv
from embedding_generator import  qdrant_client
from bedrock import get_bedrock_embedding,get_llm_response

def get_rag_response(user_input: str) -> str:
    """
    Takes user input, retrieves relevant documents using the RAG chain,
    and returns the generated response.
    """
    embedding = get_bedrock_embedding(user_input)
    result = qdrant_client.query_points(
        collection_name="internet_data",
        query=embedding,
        limit=5
    )
    
    db_response = ""
    for point in result.points:
        text = point.payload.get('text', '')
        db_response += text + "\n"
    
    if not db_response.strip():
        return "I couldn't find relevant information in the provided content."
    
    sys_prompt = f"""
    You are a helpful AI assistant. Answer the user's query solely based on the provided context. 
    If the answer is not in the content, politely state that it is not provided.
    
    **Context**
    
    {db_response}
    """
    
    return get_llm_response(sys_prompt, user_input)

def log_interaction(question: str, answer: str):
    """Logs the user query and response in both a text file and a CSV file."""
    with open("log.txt", "a") as txt_file:
        txt_file.write(f"Q: {question}\nA: {answer}\n{'-'*40}\n")
    
    with open("log.csv", "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([question, answer])

st.title("Chatbot ")

user_query = st.text_input("Ask me anything about special forces :")

if st.button("Submit") and user_query:
    answer = get_rag_response(user_query)
    st.write("**Response :**", answer)
    
    # Log question and answer
    log_interaction(user_query, answer)