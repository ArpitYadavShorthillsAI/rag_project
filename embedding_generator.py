import os
import re
import time
import concurrent.futures
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from google import genai
from google.genai import errors
from bedrock import get_bedrock_embedding

load_dotenv()

qdrant_client = QdrantClient(
    url="https://94cdd9e1-5dca-439d-bfbd-d37c8311a0f2.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7EtqJGNUBBsMZx0bWFLxasqVtJxmMOwOSsx_RfJQIIc",
    timeout=60.0 
)

client = genai.Client(api_key="AIzaSyAGLmEv6-yjGztuwu3aoj-eLevzF2a4w4Q")



def split_text_into_chunks(text: str, chunk_size: int = 1000,chunk_overlap =250) -> list:
    """
    Splits the input text into chunks roughly 'chunk_size' characters long.
    Splitting occurs at full stops (periods) to keep whole sentences.
    """
    # Split text into sentences (assumes sentences end with a period).
    sentences = re.split(r'(?<=\.)\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding the sentence would exceed the chunk size.
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_file(file_path: str) -> list:
    """
    Reads a file, splits its content into chunks, and computes embeddings for each chunk.
    Returns a list of dictionaries with the chunk text, its embedding, and the source file.
    """
    points_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = split_text_into_chunks(text)
    time.sleep(1)  # Simulate processing time
    for chunk in chunks:
        embedding = get_bedrock_embedding(chunk)
        points_data.append({
            "chunk": chunk,
            "embedding": embedding,
            "source": file_path
        })
    return points_data

def process_file_with_retries(file_path: str, retries: int = 5, delay: float = 10.0) -> list:
    """
    Wrapper for process_file with a retry mechanism.
    Attempts to process the file up to 'retries' times if exceptions occur.
    """
    for attempt in range(retries):
        try:
            return process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path} on attempt {attempt + 1}/{retries}: {e}")
            time.sleep(delay)
    print(f"Failed to process {file_path} after {retries} attempts.")
    return []

if __name__ == "__main__":
    
    qdrant_client.create_collection(
        collection_name="internet_data",
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),    
    )

    data_folder = "./data"
    file_paths = [
        os.path.join(data_folder, file)
        for file in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, file))
    ]


    all_points_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_file_with_retries, file_paths)
        for points_data in results:
            all_points_data.extend(points_data)


    print(f"Processed {len(all_points_data)} points from {len(file_paths)} files.")

    qdrant_points = []
    for idx, point_data in enumerate(all_points_data):
        qdrant_points.append(models.PointStruct(
            id=idx,
            vector=point_data["embedding"],
            payload={
                "text": point_data["chunk"],
                "source": point_data["source"]
            }
        ))
    # store these points locally
    import pickle
    with open("qdrant_points.pkl", "wb") as f:
        pickle.dump(qdrant_points, f)

    print(f"Generated {len(qdrant_points)} Qdrant points.")


      # Batch Upsert to Qdrant
    batch_size = 100  # Adjust based on performance
    for i in range(0, len(qdrant_points), batch_size):
        batch = qdrant_points[i : i + batch_size]
        qdrant_client.upsert(
            collection_name="internet_data",
            points=batch
        )
        print(f"Upserted batch {i // batch_size + 1} with {len(batch)} points.")

    print(f"Successfully upserted {len(qdrant_points)} points into Qdrant.")