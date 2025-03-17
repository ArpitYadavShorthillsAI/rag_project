from embedding_generator import get_embedding, qdrant_client
from gemini import get_llm_response
import time


query = "What is special forces of india? what are the types of special forces?"
embedding = get_embedding(query)
time.sleep(1)  

result = qdrant_client.query_points(
    collection_name="internet_data",
    query=embedding,
    limit=5
)

# print(result)

db_response = ""

for point in result.points:
    text = point.payload.get('text')
    db_response += text

sys_prompt = f"""
You are a helpful ai assistant, answer the user's query solely based on the context provided, incase you dont know the answer, politely state that it is not provided in the content.


**Context**

{db_response}
"""

prompt = query

llm_respone = get_llm_response(sys_prompt, prompt)
print(llm_respone)