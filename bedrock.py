import boto3
import json
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import botocore.exceptions
import time 
from dotenv import load_dotenv
import os
load_dotenv()

def get_bedrock_embedding(text):
    """
    Generate text embedding using AWS Bedrock's Titan Embeddings model.
    
    :param text: The input text to embed
    :param model_id: The Bedrock model ID (default: Titan Embeddings v1)
    :return: Embedding vector as a list
    """
    # Initialize AWS Bedrock client
    bedrock = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1"
    )
    
    # Prepare payload
    kwargs={
    "modelId": "amazon.titan-embed-text-v2:0",
    "contentType": "application/json",
    "accept": "*/*",
    "body":json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})

    }
    
    # Invoke model
    response = bedrock.invoke_model(
        **kwargs
    )
    # print(response)
    # Parse response
    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


# Retry decorator: Retry up to 5 times, waiting 10s between attempts if a rate limit error occurs
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(10),
    retry=retry_if_exception_type(botocore.exceptions.ClientError)
)
def get_llm_response(sys_prompt, prompt) -> str:
    # Initialize AWS Bedrock client
    bedrock = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1"
    )

    # Prepare payload
    kwargs = {
        "modelId": "amazon.nova-micro-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "inferenceConfig": {
                "max_new_tokens": 1000
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"{sys_prompt}\n\n{prompt}"
                        }
                    ]
                }
            ]
        })
    }

    try:
        # Invoke model
        response = bedrock.invoke_model(**kwargs)

        # Parse response
        response_body = json.loads(response["body"].read())
        return response_body["output"]["message"]["content"][0]["text"]
    
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] in ['ThrottlingException', 'RateLimitExceeded']:
            print("Rate limit exceeded. Retrying...")
            raise  # This triggers the retry
        else:
            raise  # Re-raise if it's another error
if __name__ == "__main__":
    text = "AWS Bedrock provides foundation models for AI applications."
    embedding = get_bedrock_embedding(text)
    print("Embedding vector:", embedding)
