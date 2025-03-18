import json
from concurrent.futures import ThreadPoolExecutor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from embedding_generator import qdrant_client
from bedrock import get_llm_response, get_bedrock_embedding

def load_golden_set(filepath):
    """Loads the golden dataset from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

def query_chatbot(question):
    """Fetches an AI-generated response for a given question."""
    embedding = get_bedrock_embedding(question)

    result = qdrant_client.query_points(
        collection_name="internet_data", query=embedding, limit=5
    )

    db_response = " ".join([point.payload.get("text", "") for point in result.points])

    sys_prompt = f"""
    You are a helpful AI assistant. Answer the user's query solely based on the provided context.
    If the answer is not in the context, say so politely.
    
    **Context**
    {db_response}
    """

    return get_llm_response(sys_prompt, question)

def calculate_bleu(reference, candidate):
    """
    Computes BLEU score with unigram precision and smoothing.
    - `reference` should be a list of lists.
    - `candidate` should be a list of words.
    """
    reference = [reference.split()]  # Convert reference to list of lists
    candidate = candidate.split()  # Convert candidate to list of words

    if not candidate or not reference[0]:  # Handle empty input cases
        return 0.0  

    # Use unigram BLEU + smoothing to avoid zero scores for short responses
    return sentence_bleu(reference, candidate, weights=(1.0, 0, 0, 0), 
                         smoothing_function=SmoothingFunction().method1)

def evaluate_responses(golden_set):
    """Evaluates chatbot responses against a golden dataset using BLEU score."""

    def process_entry(entry):
        chatbot_answer = query_chatbot(entry["question"])

        # Compute BLEU score
        bleu_score = calculate_bleu(entry["expected_answer"], chatbot_answer)

        print(f"Question: {entry['question']}\n"
              f"Expected Answer: {entry['expected_answer']}\n"
              f"Chatbot Answer: {chatbot_answer}\n"
              f"BLEU Score: {bleu_score:.4f}\n")

        return entry["question"], bleu_score

    # Limit concurrency to prevent excessive API calls
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_entry, golden_set))

    # Convert results into dictionary
    scores = {question: score for question, score in results}

    # Store evaluation results in a JSON file
    with open("evaluation_results_set3_updated.json", "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=4)

    print("Evaluation results saved to evaluation_results.json")

if __name__ == "__main__":
    golden_set = load_golden_set("golden_set_3.json")
    evaluate_responses(golden_set)
