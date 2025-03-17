from embedding_generator import client

def get_llm_response(sys_prompt,prompt) -> str:  
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=sys_prompt + prompt,
    )

    return response.text