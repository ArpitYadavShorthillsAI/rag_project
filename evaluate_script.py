import json
import time
from nltk.translate.bleu_score import sentence_bleu
from embedding_generator import get_embedding, qdrant_client
from gemini import get_llm_response

def load_golden_set(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

def query_chatbot(question):
    embedding = get_embedding(question)
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
    time.sleep(10)  # Avoid hitting API rate limits
    return get_llm_response(sys_prompt, question)

def evaluate_responses(golden_set):
    for entry in golden_set:
        entry["chatbot_answer"] = query_chatbot(entry["question"])
    
    scores = {
        entry["question"]: sentence_bleu([entry["expected_answer"].split()], entry["chatbot_answer"].split())
        for entry in golden_set
    }
    
    # Store evaluation results in a JSON file
    with open("evaluation_results_2.json", "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=4)
    
    return scores

if __name__ == "__main__":
    golden_set = load_golden_set("golden_set.json")
    scores = evaluate_responses(golden_set)
    print("Evaluation results saved to evaluation_results.json")



#print(f"Stored vector dimension: {stored_vector_dim}, Query vector dimension: {len(query_vector)}")
