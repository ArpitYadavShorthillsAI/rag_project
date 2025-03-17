# rag_project
Indian Forces and Operations Chatbot

The Indian Forces Chatbot is an AI-powered chatbot designed to answer queries about special forces and operations. It uses data scraped from Wikipedia, processes it using embeddings and vector storage, and generates responses through a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **Web Scraping:** Extracts text from Wikipedia about special forces.
- **Chunking & Storage:** Stores processed text chunks in Qdrant.
- **Embeddings:** Uses AWS Bedrock for vector embeddings.
- **RAG Pipeline:** Retrieves relevant information and generates responses using Nova Micro.
- **Test Case Evaluation:** Evaluates chatbot performance using BLEU with a golden set of 1000 test cases.

## Architecture

1. **Scraping:** Wikipedia text is extracted and stored.
2. **Processing:** Text is chunked and stored in Qdrant.
3. **Embedding:** Chunks are embedded using AWS Bedrock.
4. **Retrieval:** Relevant chunks are retrieved via similarity search.
5. **Response Generation:** Nova Micro generates responses based on retrieved data.
6. **Evaluation:** Evaluates chatbot performance using BLEU with a golden set of 1000 test cases(combined)


## Tech Stack

- **Python** (BeautifulSoup for scraping)
- **Qdrant** (Vector database for storing embeddings)
- **AWS Bedrock** (For generating embeddings)
- **Nova Micro** (For LLM-based response generation)
- **BLEU** (For performance evaluation)

