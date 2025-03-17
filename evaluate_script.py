import json
import time
from nltk.translate.bleu_score import sentence_bleu
from embedding_generator import  qdrant_client
from bedrock import get_llm_response  ,get_bedrock_embedding

from concurrent.futures import ThreadPoolExecutor
from nltk.translate.bleu_score import sentence_bleu

def load_golden_set(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

def query_chatbot(question):
    embedding = get_bedrock_embedding(question)
    result = qdrant_client.query_points(
        collection_name="internet_data", query=embedding, limit=5
    )
    
    db_response = "".join([point.payload.get('text', '') for point in result.points])
    sys_prompt = f"""
    You are a helpful AI assistant. Answer the user's query solely based on the provided context.
    If the answer is not in the context, say so politely.
    
    **Context**
    {db_response}
    """

    return get_llm_response(sys_prompt, question)

def evaluate_responses(golden_set):
    def process_entry(entry):
    
        entry["chatbot_answer"] = query_chatbot(entry["question"])
        print(f"Question: {entry['question']}\nExpected Answer: {entry['expected_answer']}\nChatbot Answer: {entry['chatbot_answer']}\n")
        return entry["question"], sentence_bleu([entry["expected_answer"].split()], entry["chatbot_answer"].split())

    # Use ThreadPoolExecutor with max_workers=5
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_entry, golden_set))

    # Convert results into dictionary
    scores = {question: score for question, score in results}

    # Store evaluation results in a JSON file
    with open("evaluation_results_set_4.json", "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=4)


if __name__ == "__main__":
    golden_set = load_golden_set("golden_set_4.json")
    scores = evaluate_responses(golden_set)
    print("Evaluation results saved to evaluation_results_set_4.json")



#print(f"Stored vector dimension: {stored_vector_dim}, Query vector dimension: {len(query_vector)}")
