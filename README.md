# rag_project
Special Forces Chatbot

The Special Forces Chatbot is an AI-powered chatbot designed to answer queries about special forces. It uses data scraped from Wikipedia, processes it using embeddings and vector storage, and generates responses through a Retrieval-Augmented Generation (RAG) pipeline.

Features

Web Scraping: Extracts text from Wikipedia about special forces.

Chunking & Storage: Stores processed text chunks in Qdrant.

Embeddings: Uses Gemini to generate vector embeddings.

RAG Pipeline: Retrieves relevant information and generates responses using Gemini.

Test Case Evaluation: Evaluates chatbot performance using Bleu with a golden set of 1000 test cases.

Architecture

Scraping: Wikipedia text is extracted and stored.

Processing: Text is chunked and stored in Qdrant.

Embedding: Chunks are embedded using Gemini.

Retrieval: Relevant chunks are retrieved via similarity search.

Response Generation: Gemini generates responses based on retrieved data.

Evaluation: Ragas assesses accuracy, relevance, and factual correctness.

Tech Stack

Python (BeautifulSoup for scraping)

Qdrant (Vector database)

Gemini (Embeddings & RAG)

BlEU (Evaluation)

Setup Instructions

Clone the repository:

git clone <repo_url>
cd special-forces-chatbot

Install dependencies:

pip install -r requirements.txt

Run the scraper:

python scraper.py

Store embeddings in Qdrant:

python embedding_generator.py

Start the chatbot:

python app.py

Evaluate performance:

python evaluate_script.py

Expand test cases for improved evaluation.

Implement real-time Wikipedia updates.

